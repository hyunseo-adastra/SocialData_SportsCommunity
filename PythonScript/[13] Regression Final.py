import pandas as pd
import numpy as np
import glob
import os
import statsmodels.formula.api as smf

# =========================================================
# 1. 경로 및 데이터 로드 (기존 로직 유지)
# =========================================================
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
event_file = os.path.join(base_path, 'EventLogData/F1 Event Data.xlsx')
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

try:
    events = pd.read_excel(event_file)
    events['unexpected'] = pd.to_numeric(events['unexpected'], errors='coerce').fillna(0).astype(int)
    events['importance'] = pd.to_numeric(events['importance'], errors='coerce').fillna(0).astype(int)
    events['lap_num'] = pd.to_numeric(events['lap'], errors='coerce')
    event_summary = events.dropna(subset=['lap_num', 'race']).groupby(['race', 'lap_num']).agg({
        'unexpected': 'max',
        'importance': 'max'
    }).reset_index()
except Exception as e:
    print(f"❌ 이벤트 로드 오류: {e}")
    exit()

races = ['Bahrain', 'Australia', 'Japan', 'Imola', 'Spain', 'United_Kingdom',
         'Hungary', 'Netherlands', 'Singapore', 'Las_Vegas', 'Qatar', 'Abu_Dhabi']

all_data = []
for race in races:
    e_file = os.path.join(emo_path, f'Community_{race}_Emotion_Scores.csv')
    t_file = os.path.join(tox_path, f'Community_{race}_Toxicity_Scores.csv')
    if os.path.exists(e_file) and os.path.exists(t_file):
        df_e = pd.read_csv(e_file)
        df_t = pd.read_csv(t_file)
        df_race = pd.concat([df_e, df_t[['Toxicity_Score']]], axis=1)
        df_race['race'] = race
        df_race['post_timestamp'] = pd.to_datetime(df_race['post_timestamp'])
        df_race = df_race.sort_values('post_timestamp')
        diff_prev = df_race['post_timestamp'].diff().dt.total_seconds()
        diff_next = df_race['post_timestamp'].diff(-1).abs().dt.total_seconds()
        df_race['custom_interval'] = (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2
        df_race = pd.merge(df_race, event_summary, left_on=['race', 'LapNumber'], right_on=['race', 'lap_num'], how='left')
        all_data.append(df_race)

df_total = pd.concat(all_data, axis=0, ignore_index=True)
df_total = df_total.dropna(subset=['LapNumber']).copy() # 경기 중 데이터 필터링

# 변수 생성
df_total['unexpected'] = df_total['unexpected'].fillna(0).astype(int)
df_total['importance'] = df_total['importance'].fillna(0).astype(int)
df_total['lap_norm'] = df_total.groupby('race')['LapNumber'].transform(lambda x: x / (x.max() if x.max() > 0 else 1))
df_total['is_toxic'] = (df_total['Toxicity_Score'] >= 0.5).astype(int)
df_total['comment_len'] = df_total['processed_text'].astype(str).apply(len)

# =========================================================
# 5. 회귀 분석 및 출력 (4대 감정 포함)
# =========================================================
def print_paper_format(model_res, title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(model_res.summary().tables[1])
    r2 = getattr(model_res, 'rsquared', getattr(model_res, 'prsquared', 'N/A'))
    print(f"R-squared: {r2:.3f}" if isinstance(r2, float) else f"R-squared: {r2}")
    print(f"N: {int(model_res.nobs):,}")

# (1) 댓글 간격 분석
res_interval = smf.ols('custom_interval ~ unexpected + importance + lap_norm + comment_len', data=df_total).fit()
print_paper_format(res_interval, "표 n. 종속변수: 댓글 간격 (종목 F)")

# (2) 4대 주요 감정 선형 회귀 (Anger, Happiness, Disgust, Surprise)
# LABEL_0: Anger, LABEL_3: Happiness, LABEL_1: Disgust, LABEL_5: Surprise
target_emotions = [
    ('Anger', 'LABEL_0'),
    ('Happiness', 'LABEL_3'),
    ('Disgust', 'LABEL_1'),
    ('Surprise', 'LABEL_5')
]

for name, col in target_emotions:
    res_emo = smf.ols(f'{col} ~ unexpected + importance + lap_norm', data=df_total).fit()
    print_paper_format(res_emo, f"종목 F {name} 감정 분석 결과")

# (3) 독성 로지스틱 회귀
res_toxic = smf.logit('is_toxic ~ unexpected + importance + lap_norm + comment_len', data=df_total).fit()
print_paper_format(res_toxic, "종목 F 독성 분석 결과 (Logit)")
print("\n--- Odds Ratios (오즈비) ---")
print(np.exp(res_toxic.params))