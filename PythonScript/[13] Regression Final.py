import pandas as pd
import numpy as np
import glob
import os
import statsmodels.formula.api as smf


# =========================================================
# 0. 출력 함수 정의 (논문 양식 맞춤)
# =========================================================
def print_paper_format(model_res, title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    # 회귀 분석 결과 테이블 출력
    print(model_res.summary().tables[1])

    # R-squared 계산 (선형 회귀는 rsquared, 로지스틱은 prsquared)
    if hasattr(model_res, 'rsquared'):
        r2 = model_res.rsquared
        r2_label = "R-squared"
    elif hasattr(model_res, 'prsquared'):
        r2 = model_res.prsquared
        r2_label = "Pseudo R-squared"
    else:
        r2 = "N/A"
        r2_label = "R-squared"

    if isinstance(r2, float):
        print(f"{r2_label}: {r2:.3f}")
    else:
        print(f"{r2_label}: {r2}")

    # 표본 수 출력
    print(f"N (표본수): {int(model_res.nobs):,}")

# =========================================================
# 1. 경로 및 데이터 로드
# =========================================================
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
event_file = os.path.join(base_path, 'EventLogData/F1 Event Data.xlsx')
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

# 2024 시즌 스케줄 순서 (경기의 중요도 변수 산출용)
race_order = {
    'Bahrain': 1, 'Australia': 3, 'Japan': 4, 'Imola': 7,
    'Spain': 10, 'United_Kingdom': 12, 'Hungary': 13, 'Netherlands': 15,
    'Singapore': 18, 'Las_Vegas': 22, 'Qatar': 23, 'Abu_Dhabi': 24
}

try:
    events = pd.read_excel(event_file)
    events.columns = events.columns.str.strip()
    events['unexpected'] = pd.to_numeric(events['unexpected'], errors='coerce').fillna(0).astype(int)
    events['importance'] = pd.to_numeric(events['importance'], errors='coerce').fillna(0).astype(int)
    events['lap_num'] = pd.to_numeric(events['lap'], errors='coerce')

    event_summary = events.dropna(subset=['lap_num', 'race']).groupby(['race', 'lap_num']).agg({
        'unexpected': 'max',
        'importance': 'max'
    }).reset_index()
    print("✅ 이벤트 로그 데이터 로드 완료")
except Exception as e:
    print(f"❌ 이벤트 로드 오류: {e}")
    exit()

# 분석 대상 레이스
races = list(race_order.keys())

all_data = []
for race in races:
    e_file = os.path.join(emo_path, f'Community_{race}_Emotion_Scores.csv')
    t_file = os.path.join(tox_path, f'Community_{race}_Toxicity_Scores.csv')

    if os.path.exists(e_file) and os.path.exists(t_file):
        df_e = pd.read_csv(e_file)
        df_t = pd.read_csv(t_file)
        df_race = pd.concat([df_e, df_t[['Toxicity_Score']]], axis=1)
        df_race['race'] = race

        # 시간 간격 계산
        df_race['post_timestamp'] = pd.to_datetime(df_race['post_timestamp'])
        df_race = df_race.sort_values('post_timestamp')
        diff_prev = df_race['post_timestamp'].diff().dt.total_seconds()
        diff_next = df_race['post_timestamp'].diff(-1).abs().dt.total_seconds()
        df_race['custom_interval'] = (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2

        # 이벤트 매칭
        df_race = pd.merge(df_race, event_summary, left_on=['race', 'LapNumber'], right_on=['race', 'lap_num'],
                           how='left')
        all_data.append(df_race)

df_total = pd.concat(all_data, axis=0, ignore_index=True)
df_total = df_total.dropna(subset=['LapNumber']).copy()

# =========================================================
# 2. 통제 변수 및 독립 변수 확정 (논문 3.3절 반영)
# =========================================================
# (1) 경기의 중요도 (시즌 내 시점 반영: 0~1)
race_order = {
    'Bahrain': 1, 'Australia': 3, 'Japan': 4, 'Imola': 7,
    'Spain': 10, 'United_Kingdom': 12, 'Hungary': 13, 'Netherlands': 15,
    'Singapore': 18, 'Las_Vegas': 22, 'Qatar': 23, 'Abu_Dhabi': 24
}
df_total['season_progression'] = df_total['race'].map(race_order) / 24

# (2) 경기 내 시간 (in-game timestamp: 0~1)
df_total['lap_norm'] = df_total.groupby('race')['LapNumber'].transform(lambda x: x / (x.max() if x.max() > 0 else 1))

# (3) 경기 시간대 (한국 시간 기준 - 야간 경기 여부 등 더미화 가능)
# 단순하게는 시작 시간(Hour)을 통제로 넣거나, 특정 시간대 그룹화 가능
df_total['hour'] = df_total['post_timestamp'].dt.hour

# (4) 독립 변수 및 종속 변수 이진화
df_total['unexpected'] = df_total['unexpected'].fillna(0).astype(int)
df_total['importance'] = df_total['importance'].fillna(0).astype(int)
df_total['is_toxic'] = (df_total['Toxicity_Score'] >= 0.5).astype(int)

# =========================================================
# 3. 회귀 분석 실행 (표 6, 7, 8 작성용)
# =========================================================

# (1) RQ1: 댓글 간격 분석 (통제 변수에서 comment_len 제외)
# 논문 본문 서술대로 '중요도', '진행도', '시간대'를 통제 변수로 포함
res_interval = smf.ols(
    'custom_interval ~ unexpected + importance + season_progression + lap_norm + hour',
    data=df_total
).fit()
print_paper_format(res_interval, "표 6. 종속변수: 댓글 간격 (종목 F)")

# (2) RQ2: 4대 감정 분석
target_emotions = [('Anger', 'LABEL_0'), ('Happiness', 'LABEL_3'), ('Disgust', 'LABEL_1'), ('Surprise', 'LABEL_5')]
for name, col in target_emotions:
    res_emo = smf.ols(f'{col} ~ unexpected + importance + season_progression + lap_norm + hour', data=df_total).fit()
    print_paper_format(res_emo, f"종목 F {name} 감정 분석 결과")

# (3) RQ3: 독성 분석 (로지스틱)
res_toxic = smf.logit('is_toxic ~ unexpected + importance + season_progression + lap_norm + hour', data=df_total).fit()
print_paper_format(res_toxic, "종목 F 독성 분석 결과")
print("\n--- Odds Ratios ---")
print(np.exp(res_toxic.params))