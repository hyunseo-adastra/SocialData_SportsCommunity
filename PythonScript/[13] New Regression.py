import pandas as pd
import numpy as np
import glob
import os
import statsmodels.formula.api as smf

# 1. 경로 설정
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

# 2. 레이스 리스트 (실제 파일이 있는 레이스들)
races = ['United_Kingdom', 'Hungary', 'Japan', 'Abu_Dhabi']  # 분석할 레이스 추가

all_race_data = []

print("🔄 데이터 통합 및 변수 계산 중...")

for race in races:
    try:
        # 파일 로드
        e_file = os.path.join(emo_path, f'Community_{race}_Emotion_Scores.csv')
        t_file = os.path.join(tox_path, f'Community_{race}_Toxicity_Scores.csv')

        if not os.path.exists(e_file) or not os.path.exists(t_file):
            continue

        df_emo = pd.read_csv(e_file)
        df_tox = pd.read_csv(t_file)

        # 감정 데이터와 독성 데이터 합치기 (Toxicity_Score만 가져옴)
        df_race = pd.concat([df_emo, df_tox[['Toxicity_Score']]], axis=1)
        df_race['race'] = race

        # 🌟 핵심: custom_interval 계산 (타임스탬프 기반)
        df_race['post_timestamp'] = pd.to_datetime(df_race['post_timestamp'])
        df_race = df_race.sort_values('post_timestamp')


        def calc_interval(group):
            diff_prev = group['post_timestamp'].diff().dt.total_seconds()
            diff_next = group['post_timestamp'].diff(-1).abs().dt.total_seconds()
            return (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2


        df_race['custom_interval'] = calc_interval(df_race)

        all_race_data.append(df_race)
        print(f" ✅ {race} 처리 완료")

    except Exception as e:
        print(f" ❌ {race} 처리 중 오류: {e}")

# 전체 데이터 통합
df = pd.concat(all_race_data, axis=0, ignore_index=True)

# 3. 독립 변수(의외성, 중요성) 코딩 (본인의 랩 번호 리스트로 수정)
# 예시 랩 번호 (본인의 분석 기준에 맞춰 수정하세요)
unpredictable_laps = {'United_Kingdom': [15, 20], 'Hungary': [5, 45], 'Japan': [12], 'Abu_Dhabi': [50]}
important_laps = {'United_Kingdom': [1, 52], 'Hungary': [1, 70], 'Japan': [1, 53], 'Abu_Dhabi': [1, 58]}

df['unpredictability'] = df.apply(
    lambda x: 1 if x['race'] in unpredictable_laps and x['LapNumber'] in unpredictable_laps[x['race']] else 0, axis=1)
df['importance'] = df.apply(
    lambda x: 1 if x['race'] in important_laps and x['LapNumber'] in important_laps[x['race']] else 0, axis=1)

# 독성 이진화 및 경기 시간 정규화
df['is_toxic'] = (df['Toxicity_Score'] >= 0.5).astype(int)
df['lap_norm'] = df['LapNumber'].fillna(0) / 70  # 대략적인 최대 랩으로 정규화

# =========================================================
# 4. 회귀 분석 실행 (표의 빈칸 채우기용)
# =========================================================

# (1) 댓글 간격 (RQ1)
print("\n[표 n. 종목 F 댓글 간격 선형 회귀]")
model_interval = smf.ols('custom_interval ~ unpredictability + importance + lap_norm', data=df).fit()
print(model_interval.summary().tables[1])

# (2) 댓글 감정 (RQ2) - Happiness(LABEL_3)와 Anger(LABEL_0) 예시
for emo_name, col in [('Anger', 'LABEL_0'), ('Happiness', 'LABEL_3')]:
    print(f"\n[표 n. 종목 F {emo_name} 감정 선형 회귀]")
    model_emo = smf.ols(f'{col} ~ unpredictability + importance + lap_norm', data=df).fit()
    print(model_emo.summary().tables[1])

# (3) 댓글 독성 (RQ3) - 로지스틱
print("\n[표 n. 종목 F 독성 로지스틱 회귀]")
model_toxic = smf.logit('is_toxic ~ unpredictability + importance + lap_norm', data=df).fit()
print(model_toxic.summary().tables[1])
print("\n--- Odds Ratios ---")
print(np.exp(model_toxic.params))