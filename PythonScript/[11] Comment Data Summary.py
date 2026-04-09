import pandas as pd
import numpy as np

# =========================
# 0. 경로
# =========================
analysis_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/AnalysisResults/f1_race_lap_analysis.xlsx'
event_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EventLogData/F1 Event Data.xlsx'
comment_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EmotionAnalysis/f1_community_prediction_result.csv'

output_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/AnalysisResults/f1_race_lap_analysis_with_event.xlsx'

# =========================
# 1. 로드
# =========================
print("데이터 로드 중...")
df_analysis = pd.read_excel(analysis_path)
df_event = pd.read_excel(event_path)
df_c = pd.read_csv(comment_path)

# 공백 제거
df_analysis.columns = df_analysis.columns.str.strip()
df_event.columns = df_event.columns.str.strip()
df_c.columns = df_c.columns.str.strip()

# =========================
# 2. 공통 lap key 함수 (그대로)
# =========================
def make_common_lap_key(val):
    s = str(val).strip().lower()

    if s == 'finish':
        return 'After Lap'
    if s == 'after lap':
        return 'After Lap'

    clean_s = s.replace('lap', '').strip()
    try:
        num = int(float(clean_s))
        return f"Lap {num}"
    except:
        return str(val).strip()

# =========================
# 3. 댓글 데이터에서 avg_time_gap 계산 (네 방식)
# =========================
print("\n[댓글 단위 간격 계산 및 race-lap 집계 중...]")

# ✅ 댓글 데이터 컬럼명 자동 탐색 (필요시 여기만 고쳐)
# race
race_col = next((c for c in df_c.columns if c.lower() == 'race' or 'race' in c.lower()), None)
if race_col is None:
    raise ValueError("댓글 데이터에서 race 컬럼을 찾지 못했어요")

# lap
lap_col = next((c for c in df_c.columns if c.lower() == 'lap' or 'lap' in c.lower()), None)
if lap_col is None:
    raise ValueError("댓글 데이터에서 lap 컬럼을 찾지 못했어요")

# timestamp
ts_col = next((c for c in df_c.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'created' in c.lower()), None)
if ts_col is None:
    raise ValueError("댓글 데이터에서 timestamp 컬럼을 찾지 못했어요")

# text
text_col = next((c for c in df_c.columns if c.lower() == 'text' or 'text' in c.lower() or 'comment' in c.lower()), None)
if text_col is None:
    raise ValueError("댓글 데이터에서 text 컬럼을 찾지 못했어요")

# 기본 정리
df_c['race'] = df_c[race_col].astype(str).str.strip()
df_c['join_lap'] = df_c[lap_col].apply(make_common_lap_key)

df_c[ts_col] = pd.to_datetime(df_c[ts_col], errors='coerce')
df_c = df_c.dropna(subset=[ts_col])

# 텍스트 길이(공백 제외 글자수)
df_c['text_len_nospace'] = (
    df_c[text_col]
    .astype(str)
    .str.replace(r'\s+', '', regex=True)
    .str.len()
)

# race-lap 내부에서 시간정렬
df_c = df_c.sort_values(['race', 'join_lap', ts_col])

prev_t = df_c.groupby(['race', 'join_lap'])[ts_col].shift(1)
next_t = df_c.groupby(['race', 'join_lap'])[ts_col].shift(-1)

gap_prev = (df_c[ts_col] - prev_t).dt.total_seconds()
gap_next = (next_t - df_c[ts_col]).dt.total_seconds()

# 중간댓글: (t_{i+1} - t_{i-1}) / 2  == (gap_prev + gap_next)/2
mid_gap = (gap_prev + gap_next) / 2.0

# 첫댓글: 다음댓글과의 간격, 마지막댓글: 이전댓글과의 간격
df_c['comment_gap'] = np.where(
    prev_t.isna(), gap_next,
    np.where(next_t.isna(), gap_prev, mid_gap)
)

# 음수/이상치 정리(선택)
df_c.loc[df_c['comment_gap'] < 0, 'comment_gap'] = np.nan

# race-lap 단위 집계
agg = (
    df_c.groupby(['race', 'join_lap'], as_index=False)
        .agg(
            comment_count=(text_col, 'count'),
            avg_text_len=('text_len_nospace', 'mean'),
            avg_time_gap=('comment_gap', 'mean'),
        )
)

# =========================
# 4. df_analysis에 댓글 집계 merge
# =========================
print("\n[analysis에 댓글 집계 merge 중...]")

df_analysis['race'] = df_analysis['race'].astype(str).str.strip()
df_analysis['join_lap'] = df_analysis['lap'].apply(make_common_lap_key)

# 기존 집계 컬럼이 이미 있으면 제거해서 merge 시 _x/_y suffix가 생기지 않게 함
for c in ["comment_count", "avg_text_len", "avg_time_gap"]:
    if c in df_analysis.columns:
        df_analysis = df_analysis.drop(columns=[c])

df_analysis = df_analysis.merge(
    agg,
    on=['race', 'join_lap'],
    how='left'
)

# join_lap은 이벤트 merge에도 쓸 거라 아직 유지

# =========================
# 5. 이벤트 merge (네 코드 거의 그대로)
# =========================
print("\n[이벤트 매칭 키 생성 중...]")

df_event['race'] = df_event['race'].astype(str).str.strip()

event_lap_col = 'Lap'
for col in df_event.columns:
    if 'lap' in col.lower():
        event_lap_col = col
        break

df_event['join_lap'] = df_event[event_lap_col].apply(make_common_lap_key)

print("\n--- 키 포맷 확인 (상위 5개) ---")
print(f"Analysis 쪽 키: {df_analysis['join_lap'].unique()[:5]}")
print(f"Event 쪽 키   : {df_event['join_lap'].unique()[:5]}")

common_keys = set(df_analysis['join_lap']).intersection(set(df_event['join_lap']))
print(f"\n매칭 가능한 공통 Lap 개수: {len(common_keys)} 종류")

cols_to_use = ['race', 'join_lap', 'ev_unexp', 'ev_resp', 'ev_out']
merged_df = pd.merge(
    df_analysis,
    df_event[cols_to_use],
    on=['race', 'join_lap'],
    how='left'
)

merged_df['event_marker'] = np.where(
    merged_df[['ev_unexp', 'ev_resp', 'ev_out']].notna().any(axis=1),
    1, 0
)

for c in ['ev_unexp', 'ev_resp', 'ev_out']:
    merged_df[c] = merged_df[c].fillna(0).astype(int)

# join_lap 제거
merged_df = merged_df.drop(columns=['join_lap'])

# =========================
# 6. 저장
# =========================
merged_df.to_excel(output_path, index=False)

print("-" * 30)
print("작업 완료!")
print(f"총 데이터 개수: {len(merged_df)}")
print(f"이벤트 매칭 성공(marker=1) 개수: {merged_df['event_marker'].sum()}")

print("\n[댓글 집계 결측치 체크]")
# avg_time_gap 컬럼명 안전 처리 (혹시 suffix가 붙는 경우 대비)
if "avg_time_gap" in merged_df.columns:
    gap_col = "avg_time_gap"
elif "avg_time_gap_y" in merged_df.columns:
    gap_col = "avg_time_gap_y"
elif "avg_time_gap_x" in merged_df.columns:
    gap_col = "avg_time_gap_x"
else:
    gap_candidates = [c for c in merged_df.columns if "time_gap" in c.lower()]
    gap_col = gap_candidates[0] if gap_candidates else None

if gap_col is None:
    print("avg_time_gap 관련 컬럼이 없습니다. 댓글 집계 merge 여부를 확인하세요")
else:
    print(f"avg_time_gap 컬럼: {gap_col}")
    print("avg_time_gap NA 비율:", merged_df[gap_col].isna().mean())