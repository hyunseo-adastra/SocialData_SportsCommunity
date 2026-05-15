import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf


# =========================================================
# 0. 출력 함수 정의 (논문 양식 맞춤)
# =========================================================
def print_paper_format(model_res, title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(model_res.summary().tables[1])

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

    print(f"N (표본수): {int(model_res.nobs):,}")


# =========================================================
# 1. 데이터 로드 및 기본 전처리
# =========================================================
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
event_file = os.path.join(base_path, 'EventLogData/F1 Event Data.xlsx')
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

race_order = {
    'Bahrain': 1, 'Australia': 3, 'Japan': 4, 'Imola': 7,
    'Spain': 10, 'United_Kingdom': 12, 'Hungary': 13, 'Netherlands': 15,
    'Singapore': 18, 'Las_Vegas': 22, 'Qatar': 23, 'Abu_Dhabi': 24
}

try:
    events = pd.read_excel(event_file)
    events.columns = events.columns.str.strip()
    event_summary = events.dropna(subset=['lap', 'race']).groupby(['race', 'lap']).agg({
        'unexpected': 'max',
        'importance': 'max'
    }).reset_index()
except Exception as e:
    print(f"❌ 데이터 로드 오류: {e}")
    exit()

all_data = []
for race in race_order.keys():
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

        df_race = pd.merge(df_race, event_summary, left_on=['race', 'LapNumber'], right_on=['race', 'lap'], how='left')
        all_data.append(df_race)

df_total = pd.concat(all_data, axis=0, ignore_index=True)
df_total = df_total.dropna(subset=['LapNumber']).copy()

# =========================================================
# 2. 통제 변수 및 독립/종속 변수 이진화
# =========================================================

# (1) 시즌 후반부 여부
df_total['season_prog'] = df_total['race'].map(race_order) / 24
df_total['is_late_season'] = (df_total['season_prog'] > 0.5).astype(int)

# (2) 경기 후반부 여부
df_total['lap_norm'] = df_total.groupby('race')['LapNumber'].transform(lambda x: x / (x.max() if x.max() > 0 else 1))
df_total['is_late_lap'] = (df_total['lap_norm'] >= 0.7).astype(int)

# (3) 독립 변수 및 종속 변수 이진화
df_total['unexpected'] = df_total['unexpected'].fillna(0).astype(int)
df_total['importance'] = df_total['importance'].fillna(0).astype(int)
df_total['is_toxic'] = (df_total['Toxicity_Score'] >= 0.5).astype(int)

# =========================================================
# 3. 회귀 분석 실행 (is_prime_time 제외)
# =========================================================

# 프라임타임 변수를 제외한 분석 식
formula_str = 'unexpected + importance + is_late_season + is_late_lap'

# (1) RQ1: 댓글 간격 분석 (OLS)
res_interval = smf.ols(f'custom_interval ~ {formula_str}', data=df_total).fit()
print_paper_format(res_interval, "표 6. 종속변수: 댓글 간격 (종목 F)")

# (2) RQ2: 4대 감정 분석 (OLS)
target_emotions = [('Anger', 'LABEL_0'), ('Happiness', 'LABEL_3'), ('Disgust', 'LABEL_1'), ('Surprise', 'LABEL_5')]
for name, col in target_emotions:
    res_emo = smf.ols(f'{col} ~ {formula_str}', data=df_total).fit()
    print_paper_format(res_emo, f"종목 F {name} 감정 분석 결과")

# (3) RQ3: 독성 분석 (Logit)
res_toxic = smf.logit(f'is_toxic ~ {formula_str}', data=df_total).fit()
print_paper_format(res_toxic, "종목 F 독성 분석 결과")

print("\n--- Odds Ratios (독성 분석) ---")
print(np.exp(res_toxic.params))

# =========================================================
# 4. 주요 변수 기술통계량 출력 (is_prime_time 제외)
# =========================================================

# 분석 변수군 정의 (hour 및 prime_time 제외)
independent_vars = ['unexpected', 'importance']
control_vars_cont = ['season_prog', 'lap_norm']
control_vars_bin = ['is_late_season', 'is_late_lap']
dependent_vars = ['custom_interval', 'LABEL_0', 'LABEL_3', 'LABEL_1', 'LABEL_5', 'Toxicity_Score']

all_analysis_vars = independent_vars + control_vars_cont + control_vars_bin + dependent_vars

label_mapping = {
    'unexpected': '사건의 의외성 (1=Yes)',
    'importance': '결과 영향성 (1=Yes)',
    'season_prog': '시즌 진행도 (0~1)',
    'lap_norm': '경기 내 시간 (0~1)',
    'is_late_season': '시즌 후반부 여부 (1=Yes)',
    'is_late_lap': '경기 후반부 여부 (1=Yes)',
    'custom_interval': '댓글 간격 (sec)',
    'LABEL_0': '분노 점수 (Anger)',
    'LABEL_3': '행복 점수 (Happiness)',
    'LABEL_1': '혐오 점수 (Disgust)',
    'LABEL_5': '놀람 점수 (Surprise)',
    'Toxicity_Score': '독성 점수 (Toxicity)'
}

desc_stats = df_total[all_analysis_vars].describe().T
desc_stats = desc_stats[['count', 'mean', 'std', 'min', 'max']]
desc_stats.index = desc_stats.index.map(label_mapping)

print(f"\n{'=' * 90}")
print(f"{'[표] 주요 변수 기술통계량 (Descriptive Statistics)':^90}")
print(f"{'=' * 90}")
print(f"{'변수명':<35} {'N':>10} {'평균':>8} {'표준편차':>8} {'최솟값':>8} {'최댓값':>8}")
print(f"{'-' * 90}")

for idx, row in desc_stats.iterrows():
    print(f"{idx:<35} {int(row['count']):>10,} {row['mean']:>10.3f} {row['std']:>10.3f} {row['min']:>10.3f} {row['max']:>10.3f}")

print(f"{'=' * 90}")

print("\n[참고] 주요 독립/통제 변수 간 상관계수 (Correlation Matrix)")
corr_matrix = df_total[independent_vars + control_vars_bin].corr()
print(corr_matrix.round(3))

# =========================================================
# 5. 이진(Dummy) 통제 변수 빈도 분석 (is_prime_time 제외)
# =========================================================

binary_vars = [
    'unexpected', 'importance',
    'is_late_season', 'is_late_lap',
    'is_toxic'
]

binary_labels = {v: label_mapping.get(v, v) for v in binary_vars}

print(f"\n{'=' * 75}")
print(f"{'[표] 이진 변수 빈도 분포 (Binary Variable Distribution)':^75}")
print(f"{'=' * 75}")
print(f"{'변수명':<30} {'구분':<10} {'빈도(N)':>12} {'비율(%)':>12}")
print(f"{'-' * 75}")

for var in binary_vars:
    counts = df_total[var].value_counts().sort_index()
    total = len(df_total)

    for val in [0, 1]:
        count = counts.get(val, 0)
        percentage = (count / total) * 100
        val_label = "0 (No)" if val == 0 else "1 (Yes)"

        display_name = binary_labels[var] if val == 0 else ""
        print(f"{display_name:<30} {val_label:<10} {count:>12,} {percentage:>11.1f}%")
    print(f"{'-' * 75}")

print(f"전체 분석 대상 데이터(N): {len(df_total):,}건")
print(f"{'=' * 75}")