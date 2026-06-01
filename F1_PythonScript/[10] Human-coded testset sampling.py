import pandas as pd
import numpy as np
import re
import os

# =============================================================================
# 1. [설정] 클리닝 및 정규화 함수 정의
# =============================================================================
TEXT_CAND = ["text", "content", "message", "body", "comment", "chat", "post_content", "post_text", "desc",
             "description"]
TITLE_CAND = ["title", "post_title", "subject", "headline"]
TIME_CAND = ["timestamp", "time", "created_at", "createdAt", "date", "datetime", "post_timestamp", "post_time",
             "post_date", "published_at", "created"]
GROUP_CAND = ["team", "team_name", "driver", "player", "user", "author", "username"]

# DC Official App 제거 정규식
_DC_OFFICIAL_PAT = re.compile(
    r"""
    ^\s*
    [\-\–\—\u2013\u2014]*\s*
    ["']?\s*
    d\s*c\s*o\s*f\s*f\s*i\s*c\s*i\s*a\s*l\s*a\s*p\s*p
    \s*["']?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE
)


def pick_col(cols, cand_list):
    """컬럼명 자동 탐색"""
    cols_set = set(cols)
    for c in cand_list:
        if c in cols_set: return c
    for col in cols:
        for k in cand_list:
            if k.lower() in col.lower(): return col
    return None


def pick_text_columns(cols):
    text_col = pick_col(cols, TEXT_CAND)
    title_col = pick_col(cols, TITLE_CAND)
    return text_col, title_col


def remove_dc_official_app_lines(s: str) -> str:
    lines = re.split(r"[\r\n]+", s)
    kept = [ln for ln in lines if not _DC_OFFICIAL_PAT.match(ln)]
    return "\n".join(kept).strip()


def normalize_text(s: str) -> str:
    """텍스트 정규화 메인 함수"""
    if not isinstance(s, str): return ""

    # 1. DC App 서명 제거
    s = remove_dc_official_app_lines(s)
    # 2. URL, 태그 제거
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    # 3. 공백 정리
    s = re.sub(r"\s+", " ", s).strip()
    return s


# =============================================================================
# 2. [실행] 데이터 로드 및 샘플링
# =============================================================================
base_path = "/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData"
races = [
    "China", "Austria"
]

SAMPLES_PER_FILE = 200
sampled_dfs = []

print(f"작업 시작: {base_path} 에서 데이터를 불러옵니다...\n")

for race in races:
    file_name = f"Community_{race}.csv"
    file_path = os.path.join(base_path, file_name)

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)

            if len(df) < SAMPLES_PER_FILE:
                current_sample = df.copy()
            else:
                current_sample = df.sample(n=SAMPLES_PER_FILE, random_state=42)

            current_sample['Source_Race'] = race
            sampled_dfs.append(current_sample)
            print(f"[{race}] 추출 완료: {len(current_sample)}건")

        except Exception as e:
            print(f"[{race}] 에러 발생: {e}")
    else:
        print(f"[{race}] 파일을 찾을 수 없습니다: {file_name}")

# =============================================================================
# 3. [실행] 병합 및 클리닝 적용
# =============================================================================
if sampled_dfs:
    # 1) 전체 병합
    df_all = pd.concat(sampled_dfs, ignore_index=True)

    # 2) 컬럼 자동 탐색
    TEXT_COL, TITLE_COL = pick_text_columns(df_all.columns)
    TIME_COL = pick_col(df_all.columns, TIME_CAND)

    # 3) 텍스트 합치기 (Title + Content)
    if TEXT_COL is not None and TITLE_COL is not None:
        df_all["_raw_text"] = df_all[TITLE_COL].fillna("").astype(str) + " " + df_all[TEXT_COL].fillna("").astype(str)
    elif TEXT_COL is not None:
        df_all["_raw_text"] = df_all[TEXT_COL].fillna("").astype(str)
    else:
        # 텍스트 컬럼이 없으면 에러 방지를 위해 빈 문자열 처리
        df_all["_raw_text"] = ""
        print("Warning: 본문 컬럼을 찾지 못했습니다.")

    # 4) 클리닝 함수 적용 (clean_text 생성)
    df_all["clean_text"] = df_all["_raw_text"].map(normalize_text)

    # 5) 타임스탬프 처리
    if TIME_COL is not None:
        if np.issubdtype(df_all[TIME_COL].dtype, np.number):
            # epoch 처리 (ms or s)
            median_val = df_all[TIME_COL].dropna().median()
            unit = 'ms' if median_val > 1e11 else 's'
            df_all["ts"] = pd.to_datetime(df_all[TIME_COL], unit=unit, errors="coerce", utc=True).dt.tz_localize(None)
        else:
            df_all["ts"] = pd.to_datetime(df_all[TIME_COL], errors="coerce")
    else:
        df_all["ts"] = ""

    # 6) 빈 텍스트 제거 (클리닝 후 내용이 없어진 데이터 삭제)
    df_clean = df_all[df_all["clean_text"].str.len() > 0].copy()

    # 7) 순서 섞기
    df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    # =============================================================================
    # 4. [저장] 휴먼 라벨링용 파일 생성
    # =============================================================================

    # 저장할 컬럼만 선택
    final_cols = ['clean_text', 'Source_Race', 'ts']
    final_df = df_clean[final_cols].copy()

    # 라벨링용 빈 칸 추가
    final_df.insert(0, 'Memo', '')
    final_df.insert(0, 'Label', '')

    # 저장
    output_filename = "human_labeling_dataset3.csv"
    output_path = os.path.join(base_path, output_filename)

    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"작업 완료!")
    print(f"최종 데이터 개수: {len(final_df)}개 (빈 텍스트 제거 후)")
    print(f"저장 경로: {output_path}")
    print("이제 파일을 열어 'Label' 컬럼에 코딩(0 or 1 등)을 진행하세요.")

else:
    print("불러온 데이터가 없습니다.")