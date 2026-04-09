import pandas as pd
import numpy as np
import os

# =========================================================
# 1. 파일 경로 설정
# =========================================================
base_dir = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'

# (1) Race-Lap 기본 분석 데이터 (댓글 수, 간격 등)
analysis_path = os.path.join(base_dir, 'AnalysisResults/f1_race_lap_analysis.xlsx')
# (2) 이벤트 데이터
event_path = os.path.join(base_dir, 'EventLogData/F1 Event Data.xlsx')
# (3) 감정 분석 결과 (개별 댓글 단위)
emotion_path = os.path.join(base_dir, 'EmotionAnalysis/f1_community_prediction_result.csv')

# 결과 저장 경로
output_path = os.path.join(base_dir, 'AnalysisResults/f1_comment_level_full_dataset.xlsx')

# =========================================================
# 2. 데이터 로드
# =========================================================
print("데이터 로드 중...")
df_analysis = pd.read_excel(analysis_path)
df_event = pd.read_excel(event_path)
df_emotion = pd.read_csv(emotion_path)

# 컬럼명 공백 제거
df_analysis.columns = df_analysis.columns.str.strip()
df_event.columns = df_event.columns.str.strip()
df_emotion.columns = df_emotion.columns.str.strip()


# =========================================================
# 3. 공통 키(Key) 생성 함수 (모든 데이터프레임에 적용)
# =========================================================
def make_common_lap_key(val):
    """모든 랩 표기를 'Lap N' 또는 'After Lap'으로 통일"""
    s = str(val).strip().lower()

    if s in ['finish', 'after lap']:
        return 'After Lap'

    clean_s = s.replace('lap', '').strip()
    try:
        num = int(float(clean_s))
        return f"Lap {num}"
    except:
        return str(val).strip()


print("\n[매칭 키 생성 중...]")

# (1) Analysis 데이터 키 적용
df_analysis['race'] = df_analysis['race'].astype(str).str.strip()
df_analysis['join_lap'] = df_analysis['lap'].apply(make_common_lap_key)

# (2) Event 데이터 키 적용
df_event['race'] = df_event['race'].astype(str).str.strip()
# Event 데이터의 Lap 컬럼 찾기
event_lap_col = next((c for c in df_event.columns if 'lap' in c.lower()), 'Lap')
df_event['join_lap'] = df_event[event_lap_col].apply(make_common_lap_key)

# (3) Emotion 데이터 키 적용 (집계를 위해 필요)
df_emotion['race'] = df_emotion['race'].astype(str).str.strip()
# Emotion 데이터의 Lap 컬럼 찾기
emo_lap_col = next((c for c in ['lap', 'LapNumber', 'Lap_Number', 'Lap'] if c in df_emotion.columns), None)
if emo_lap_col:
    df_emotion['join_lap'] = df_emotion[emo_lap_col].apply(make_common_lap_key)
else:
    raise ValueError("Emotion 데이터에서 Lap 컬럼을 찾을 수 없습니다.")

# =========================================================
# 3-1. (중요) Merge 키 유일성 보장: (race, join_lap)가 중복되면 댓글 행이 복제됨
#      - event 데이터는 동일 lap에 여러 이벤트가 있을 수 있으므로 max로 묶어 0/1 존재 여부로 통일
#      - analysis 데이터도 키 중복이 있으면 한 행만 남김
# =========================================================
# Event: key별 0/1 더미 존재 여부로 집계
cols_event = ['race', 'join_lap', 'ev_unexp', 'ev_resp', 'ev_out']
missing_ev = [c for c in cols_event if c not in df_event.columns]
if missing_ev:
    raise ValueError(f"Event 데이터에서 필요한 컬럼이 없습니다: {missing_ev}")

df_event_u = (
    df_event[cols_event]
    .copy()
)
for c in ['ev_unexp', 'ev_resp', 'ev_out']:
    df_event_u[c] = pd.to_numeric(df_event_u[c], errors='coerce')

df_event_u = (
    df_event_u
    .groupby(['race', 'join_lap'], as_index=False)
    .max()
)

# Analysis: 키 중복 제거 (동일 key가 여러 행이면 첫 행만 유지)
df_analysis_u = df_analysis.copy()
if 'join_lap' not in df_analysis_u.columns:
    df_analysis_u['race'] = df_analysis_u['race'].astype(str).str.strip()
    df_analysis_u['join_lap'] = df_analysis_u['lap'].apply(make_common_lap_key)

df_analysis_u = df_analysis_u.drop_duplicates(subset=['race', 'join_lap'])

# 디버그: 중복 키가 실제로 있었는지 확인
print("\n[DEBUG] key 중복 체크")
print("- df_emotion 중복키 수:", int(df_emotion.duplicated(subset=['race', 'join_lap', ts_col] if 'ts_col' in locals() and ts_col is not None else ['race', 'join_lap']).sum()))
print("- df_event (원본) 중복키 수:", int(df_event.duplicated(subset=['race', 'join_lap']).sum()))
print("- df_event_u (정리 후) 중복키 수:", int(df_event_u.duplicated(subset=['race', 'join_lap']).sum()))
print("- df_analysis (원본) 중복키 수:", int(df_analysis.duplicated(subset=['race', 'join_lap']).sum()))
print("- df_analysis_u (정리 후) 중복키 수:", int(df_analysis_u.duplicated(subset=['race', 'join_lap']).sum()))

# =========================================================
# 4. 감정 데이터 가공: 댓글 단위 라벨링 + 간격/길이 계산 (comment-level)
# =========================================================
print("감정 데이터 댓글 단위 가공 중...")

# 4-1. 감정 라벨링 (ID -> Name)
id2label = {0: "anger", 1: "disgust", 2: "fear", 3: "happiness", 4: "sadness", 5: "surprise"}

# confidence/score 컬럼 자동 탐색
score_col = next((c for c in ['pred_score', 'emo_conf', 'confidence', 'prob', 'score'] if c in df_emotion.columns), None)
label_col = next((c for c in ['pred_label', 'pred_label_id', 'label', 'emotion_target'] if c in df_emotion.columns), None)
if label_col is None:
    raise ValueError("Emotion 데이터에서 label 컬럼을 찾을 수 없습니다. (pred_label/pred_label_id/label/emotion_target)")

NEUTRAL_THRESHOLD = 0.5  # score/확률이 이 값 이하이면 neutral 처리

def get_emotion(row):
    # score 기반 neutral 처리 (있을 때만)
    if score_col is not None:
        try:
            sc = float(row[score_col])
            if pd.isna(sc) or sc <= NEUTRAL_THRESHOLD:
                return 'neutral'
        except Exception:
            return 'neutral'

    val = row[label_col]
    try:
        if isinstance(val, str) and "LABEL_" in val:
            lid = int(val.split("_")[-1])
        else:
            lid = int(float(val))
        return id2label.get(lid, 'neutral')
    except Exception:
        return 'neutral'

# 최종 감정 라벨
df_emotion['emotion_final'] = df_emotion.apply(get_emotion, axis=1)

# 4-2. 텍스트 길이(공백 제외) 계산
text_col = next((c for c in ['text', 'comment', 'content', 'body'] if c in df_emotion.columns), None)
if text_col is None:
    print("[WARN] Emotion 데이터에서 text 컬럼을 찾지 못했습니다. avg_text_len 관련 분석은 제한됩니다")
    df_emotion['text_len_nospace'] = np.nan
else:
    df_emotion['text_len_nospace'] = (
        df_emotion[text_col].astype(str).str.replace(r'\s+', '', regex=True).str.len()
    )

# 4-3. 댓글 간격(comment_gap) 계산: (앞/뒤 댓글 시각 차)/2 규칙
# timestamp 컬럼 자동 탐색
ts_col = next((c for c in ['timestamp', 'time', 'created_at', 'datetime', 'date'] if c in df_emotion.columns), None)
if ts_col is None:
    # 이름에 time/timestamp 포함된 컬럼도 탐색
    ts_col = next((c for c in df_emotion.columns if 'time' in c.lower() or 'timestamp' in c.lower() or 'created' in c.lower()), None)

if ts_col is None:
    print("[WARN] Emotion 데이터에서 timestamp 컬럼을 찾지 못했습니다. comment_gap은 NaN으로 남습니다")
    df_emotion['comment_gap'] = np.nan
else:
    df_emotion[ts_col] = pd.to_datetime(df_emotion[ts_col], errors='coerce')
    # race/join_lap 단위로 시간 정렬
    df_emotion = df_emotion.dropna(subset=[ts_col])
    df_emotion = df_emotion.sort_values(['race', 'join_lap', ts_col])

    prev_t = df_emotion.groupby(['race', 'join_lap'])[ts_col].shift(1)
    next_t = df_emotion.groupby(['race', 'join_lap'])[ts_col].shift(-1)

    gap_prev = (df_emotion[ts_col] - prev_t).dt.total_seconds()
    gap_next = (next_t - df_emotion[ts_col]).dt.total_seconds()

    mid_gap = (gap_prev + gap_next) / 2.0

    df_emotion['comment_gap'] = np.where(
        prev_t.isna(), gap_next,
        np.where(next_t.isna(), gap_prev, mid_gap)
    )

    # 음수 간격 방지
    df_emotion.loc[df_emotion['comment_gap'] < 0, 'comment_gap'] = np.nan

print(f"- 댓글 단위 감정 데이터: {len(df_emotion)}개 행")

# =========================================================
# 5. 데이터 병합 (Merge All): comment-level
# =========================================================
print("\n데이터 병합 중... (comment-level)")

# 1단계: 감정(댓글) + Event(이벤트)  [race, join_lap]
cols_event = ['race', 'join_lap', 'ev_unexp', 'ev_resp', 'ev_out']
merged_1 = pd.merge(df_emotion, df_event_u, on=['race', 'join_lap'], how='left', validate='many_to_one')

# 이벤트 마커 생성
merged_1['event_marker'] = np.where(
    merged_1[['ev_unexp', 'ev_resp', 'ev_out']].notna().any(axis=1), 1, 0
)
merged_1[['ev_unexp', 'ev_resp', 'ev_out']] = merged_1[['ev_unexp', 'ev_resp', 'ev_out']].fillna(0)

# 2단계(선택): Lap-level 분석 지표를 댓글에 붙이기 (통제변수로 유용)
# df_analysis에는 comment_count/avg_time_gap/avg_text_len 등이 들어있을 수 있음
cols_analysis = [c for c in ['race', 'join_lap', 'comment_count', 'avg_time_gap', 'avg_text_len'] if c in df_analysis_u.columns]
if len(cols_analysis) >= 2:
    merged_final = pd.merge(merged_1, df_analysis_u[cols_analysis], on=['race', 'join_lap'], how='left', suffixes=('', '_lap'), validate='many_to_one')
else:
    merged_final = merged_1

# join_lap 제거(필요하면 유지해도 됨)
merged_final = merged_final.drop(columns=['join_lap'])

print(f"\n[DEBUG] df_emotion 행수: {len(df_emotion):,}")
print(f"[DEBUG] merged_final 행수: {len(merged_final):,} (이 값이 df_emotion보다 크면 키 중복으로 복제된 것)")

# =========================================================
# 6. 저장
# =========================================================
merged_final.to_excel(output_path, index=False)

print("-" * 30)
print("작업 완료!")
print(f"최종 데이터 개수: {len(merged_final)}")
print(f"저장 경로: {output_path}")
print("\n[생성된 컬럼 목록]")
print(list(merged_final.columns))

# =========================================================
# 7. 회귀분석: 댓글 단위 (표 6–8)
# =========================================================
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit

print("\n================ 회귀분석 시작 (댓글 단위) ================")

# ---------------------------------------------------------
# 7-0. 종목 구분 컬럼 생성 (F / L)
# ---------------------------------------------------------
# race 이름 기반 규칙: LoL 키워드가 있으면 L, 그 외는 기본적으로 F로 처리
# (F1 레이스 이름은 보통 'Bahrain Grand Prix'처럼 'f1' 문자열이 없어서 기본값 F가 필요함)
def classify_sport(r):
    r = str(r).lower()
    # LoL / League
    if any(k in r for k in ['lol', 'league', 'lck', 'lpl', 'lec', 'lcs', 'msi', 'worlds']):
        return 'L'
    # 그 외는 F로 간주
    return 'F'

merged_final['sport'] = merged_final['race'].apply(classify_sport)

# 유효 표본만 사용
reg_df = merged_final.dropna(subset=['sport'])
print("사용 표본 수:")
print(reg_df['sport'].value_counts())

if len(reg_df) == 0:
    raise ValueError("sport 분류 결과가 비어 있습니다. race 값/규칙을 확인하세요")

# 공통 독립변수
X_vars = ['ev_unexp', 'ev_resp', 'ev_out']

# p-value -> significance stars
def pstars(p):
    try:
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''
    except Exception:
        return ''

# =========================================================
# [표 6] 종속변수: 댓글 간격 (OLS)
# =========================================================
print("\n[표 6] 종속변수: 댓글 간격 (OLS)")

for sp in ['F', 'L']:
    d = reg_df[(reg_df['sport'] == sp) & (~reg_df['comment_gap'].isna())]
    if len(d) == 0:
        print(f"\n--- 종목 {sp} ---")
        print("[Skip] 표본이 0개입니다")
        continue
    model = ols('comment_gap ~ ev_unexp + ev_resp + ev_out', data=d).fit()

    print(f"\n--- 종목 {sp} ---")
    print(f"N = {int(model.nobs)}")
    print(f"R2 = {model.rsquared:.3f}")
    for v in X_vars:
        p = model.pvalues.get(v, np.nan)
        st = pstars(p)
        print(f"{v}: {model.params[v]:.2f} ({model.bse[v]:.2f})  p={p:.4g}{st}")
    # constant
    p0 = model.pvalues.get('Intercept', np.nan)
    print(f"constant: {model.params['Intercept']:.2f} ({model.bse['Intercept']:.2f})  p={p0:.4g}{pstars(p0)}")

# =========================================================
# [표 7] 종속변수: 분노 (Logit, OR)
# =========================================================
print("\n[표 7] 종속변수: 댓글 감정 (분노)")

# 이진 종속변수 생성
reg_df['is_anger'] = (reg_df['emotion_final'] == 'anger').astype(int)

for sp in ['F', 'L']:
    d = reg_df[reg_df['sport'] == sp]
    if len(d) == 0:
        print(f"\n--- 종목 {sp} ---")
        print("[Skip] 표본이 0개입니다")
        continue
    model = logit('is_anger ~ ev_unexp + ev_resp + ev_out', data=d).fit(disp=False)

    print(f"\n--- 종목 {sp} ---")
    print(f"N = {int(model.nobs)}")
    print(f"-2LL = {-2 * model.llf:.0f}")
    for v in X_vars:
        or_val = np.exp(model.params[v])
        se = model.bse[v]
        p = model.pvalues.get(v, np.nan)
        st = pstars(p)
        print(f"{v}: OR={or_val:.2f} (SE={se:.2f})  p={p:.4g}{st}")
    # intercept (odds ratio)
    p0 = model.pvalues.get('Intercept', np.nan)
    or0 = float(np.exp(model.params['Intercept']))
    se0 = float(model.bse['Intercept'])
    print(f"intercept: OR={or0:.2f} (SE={se0:.2f})  p={p0:.4g}{pstars(p0)}")

# =========================================================
# [표 8] 종속변수: 행복 (Logit, OR)
# =========================================================
print("\n[표 8] 종속변수: 댓글 감정 (행복)")

reg_df['is_happy'] = (reg_df['emotion_final'] == 'happiness').astype(int)

for sp in ['F', 'L']:
    d = reg_df[reg_df['sport'] == sp]
    if len(d) == 0:
        print(f"\n--- 종목 {sp} ---")
        print("[Skip] 표본이 0개입니다")
        continue
    model = logit('is_happy ~ ev_unexp + ev_resp + ev_out', data=d).fit(disp=False)

    print(f"\n--- 종목 {sp} ---")
    print(f"N = {int(model.nobs)}")
    print(f"-2LL = {-2 * model.llf:.0f}")
    for v in X_vars:
        or_val = np.exp(model.params[v])
        se = model.bse[v]
        p = model.pvalues.get(v, np.nan)
        st = pstars(p)
        print(f"{v}: OR={or_val:.2f} (SE={se:.2f})  p={p:.4g}{st}")
    # intercept (odds ratio)
    p0 = model.pvalues.get('Intercept', np.nan)
    or0 = float(np.exp(model.params['Intercept']))
    se0 = float(model.bse['Intercept'])
    print(f"intercept: OR={or0:.2f} (SE={se0:.2f})  p={p0:.4g}{pstars(p0)}")

print("\n================ 회귀분석 종료 ================")