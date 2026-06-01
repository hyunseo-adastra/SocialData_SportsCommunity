import os
import pandas as pd

# =========================
# 1) 데이터 로드
# =========================
csv_path = "/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EmotionAnalysis/F1_emotion_predictions_kcelectra.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")
df.columns = df.columns.str.strip()

print(f"Loaded: {csv_path}")
print(f"Rows: {len(df):,}")

# =========================
# 2) 컬럼 자동 탐색 (스포츠/감정/확률)
#    - 네 데이터셋에서 스포츠 구분 컬럼명이 다를 수 있어서 후보군으로 찾음
# =========================
SPORT_COL_CANDIDATES = [
    "sport", "sports", "game", "domain", "league", "event_type",
    "sport_type", "sports_type", "category", "source",
    # 종목명이 F/L로 저장되어 있을 가능성
    "sport_code", "sports_code", "code"
]

# 감정 예측 결과 컬럼 후보
EMO_NAME_CANDIDATES = ["emo_name", "emotion", "pred_emotion", "label", "emotion_target"]
CONF_CANDIDATES = ["emo_conf", "confidence", "prob", "score"]

sport_col = next((c for c in SPORT_COL_CANDIDATES if c in df.columns), None)
emo_col = next((c for c in EMO_NAME_CANDIDATES if c in df.columns), None)
conf_col = next((c for c in CONF_CANDIDATES if c in df.columns), None)

# F1 전용 파일이라 sport 컬럼이 없을 수도 있음 -> 그 경우 전체를 종목 F로 처리
if sport_col is None:
    print("[WARN] 스포츠 구분 컬럼을 찾지 못했습니다. 전체를 종목 F로 처리합니다")
    df["sport"] = "F"
    sport_col = "sport"

if emo_col is None:
    raise ValueError(f"감정 컬럼을 찾지 못했습니다. candidates={EMO_NAME_CANDIDATES} / columns={list(df.columns)}")
if conf_col is None:
    raise ValueError(f"confidence 컬럼을 찾지 못했습니다. candidates={CONF_CANDIDATES} / columns={list(df.columns)}")

# =========================
# 3) 감정 결정 (confidence ≤ 0.5 => neutral)
# =========================
EMOTIONS = ["happiness", "anger", "surprise", "sadness", "disgust", "fear", "neutral"]

def final_label(row):
    conf = row[conf_col]
    emo = str(row[emo_col]).strip().lower()

    try:
        conf_val = float(conf)
    except Exception:
        conf_val = None

    if conf_val is None or pd.isna(conf_val) or conf_val <= 0.5:
        return "neutral"

    # 혹시 'joy', 'happy' 같은 변형이 있으면 happiness로 통일
    if emo in ["joy", "happy"]:
        emo = "happiness"

    return emo if emo in EMOTIONS else "neutral"

# numeric coercion
df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
df["emo_final"] = df.apply(final_label, axis=1)

# =========================
# 4) 종목 코드 정규화 (F / L)
#    - 데이터가 'F', 'f1', 'formula1' 등으로 들어올 수 있어서 묶어줌
# =========================
raw_sport = df[sport_col].astype(str).str.strip().str.lower()

def normalize_sport(s: str) -> str:
    # F 계열
    if s in ["f", "f1", "formula1", "formula 1", "formula-1", "f1-race", "f1race"]:
        return "F"
    # L 계열 (lol 등)
    if s in ["l", "lol", "leagueoflegends", "league of legends", "league", "lck", "lpl", "lec", "lcs"]:
        return "L"
    # 이미 F/L로 저장된 경우(대문자 포함)
    if s.upper() in ["F", "L"]:
        return s.upper()
    # 그 외는 원값을 대문자로 두되, 표에는 사용자가 원하는 것만 남길 수 있음
    return s.upper()

df["sport_norm"] = raw_sport.map(normalize_sport)

# =========================
# 5) [표 5] 종목별 댓글 감정 분포 (Count)
# =========================
# sport_norm × emo_final count
pivot = (
    df.pivot_table(index="sport_norm", columns="emo_final", aggfunc="size", fill_value=0)
    .reindex(columns=EMOTIONS, fill_value=0)
)

# 합계 컬럼
pivot["total"] = pivot.sum(axis=1)

# 표 5 순서대로 컬럼 정렬 (행복, 분노, 놀람, 슬픔, 혐오, 공포, 중립, 합계)
pivot = pivot[["happiness", "anger", "surprise", "sadness", "disgust", "fear", "neutral", "total"]]

# 표에 맞게 행 이름
# (종목 F / 종목 L만 필요하면 아래처럼 재정렬)
row_order = ["F", "L"]
existing = [r for r in row_order if r in pivot.index]
pivot = pivot.loc[existing] if existing else pivot

# 보기 좋게 한국어 컬럼명으로 변경
pivot_kr = pivot.rename(columns={
    "happiness": "행복",
    "anger": "분노",
    "surprise": "놀람",
    "sadness": "슬픔",
    "disgust": "혐오",
    "fear": "공포",
    "neutral": "중립",
    "total": "합계",
})

# 행 라벨도 '종목 F', '종목 L'로
pivot_kr.index = pivot_kr.index.map(lambda x: f"종목 {x}")

print("\n[표 5] 종목별 댓글 감정 분포")
print(pivot_kr)

# =========================
# 6) 저장 (엑셀 + CSV)
# =========================
out_dir = os.path.dirname(csv_path)
out_xlsx = os.path.join(out_dir, "Table5_emotion_distribution_by_sport.xlsx")
out_csv = os.path.join(out_dir, "Table5_emotion_distribution_by_sport.csv")

pivot_kr.to_excel(out_xlsx)
pivot_kr.to_csv(out_csv, encoding="utf-8-sig")

print("\nSaved:")
print(out_xlsx)
print(out_csv)