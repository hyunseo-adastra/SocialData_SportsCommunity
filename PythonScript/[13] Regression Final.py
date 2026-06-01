import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf


# =========================================================
# 0. 출력 함수 정의 (논문 양식 맞춤)
# =========================================================
def print_paper_format(model_res, title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    # 회귀 분석 결과 테이블 출력
    print(model_res.summary().tables[1])

    # R-squared 계산
    if hasattr(model_res, 'rsquared'):
        r2, r2_label = model_res.rsquared, "R-squared"
    elif hasattr(model_res, 'prsquared'):
        r2, r2_label = model_res.prsquared, "Pseudo R-squared"
    else:
        r2, r2_label = "N/A", "R-squared"

    if isinstance(r2, float):
        print(f"{r2_label}: {r2:.3f}")
    else:
        print(f"{r2_label}: {r2}")
    print(f"N (표본수): {int(model_res.nobs):,}")


# =========================================================
# 1. 경로 및 데이터 로드
# =========================================================
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
event_file = os.path.join(base_path, 'EventLogData/F1 Event Data.xlsx')
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

# 2024 시즌 스케줄 순서
race_order = {
    'Bahrain': 1, 'Australia': 3, 'Japan': 4, 'Imola': 7,
    'Spain': 10, 'United_Kingdom': 12, 'Hungary': 13, 'Netherlands': 15,
    'Singapore': 18, 'Las_Vegas': 22, 'Qatar': 23, 'Abu_Dhabi': 24
}

# (1) 이벤트 로그 로드
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
    print(f"❌ 이벤트 로그 로드 오류: {e}")
    exit()

# (2) 레이스별 데이터 로드 및 시간 기반 진행도 계산
all_data = []
for race in race_order.keys():
    e_file = os.path.join(emo_path, f'Community_{race}_Emotion_Scores.csv')
    t_file = os.path.join(tox_path, f'Community_{race}_Toxicity_Scores.csv')

    if os.path.exists(e_file) and os.path.exists(t_file):
        df_e = pd.read_csv(e_file)
        df_t = pd.read_csv(t_file)
        df_race = pd.concat([df_e, df_t[['Toxicity_Score']]], axis=1)
        df_race['race'] = race

        # 시간 변환
        df_race['post_timestamp'] = pd.to_datetime(df_race['post_timestamp'])
        df_race = df_race.sort_values('post_timestamp')

        # 경기 내 진행도 (inrace_progression) 계산: 경과 시간 기반 정규화 (0~1)
        start_t = df_race['post_timestamp'].min()
        end_t = df_race['post_timestamp'].max()
        duration = (end_t - start_t).total_seconds()

        if duration > 0:
            df_race['inrace_progression'] = (df_race['post_timestamp'] - start_t).dt.total_seconds() / duration
        else:
            df_race['inrace_progression'] = 0

        # 댓글 간격 계산
        diff_prev = df_race['post_timestamp'].diff().dt.total_seconds()
        diff_next = df_race['post_timestamp'].diff(-1).abs().dt.total_seconds()
        df_race['custom_interval'] = (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2

        # 이벤트 데이터 병합
        df_race = pd.merge(df_race, event_summary, left_on=['race', 'LapNumber'], right_on=['race', 'lap_num'],
                           how='left')
        all_data.append(df_race)

df_total = pd.concat(all_data, axis=0, ignore_index=True)
df_total = df_total.dropna(subset=['LapNumber']).copy()

# =========================================================
# 2. 통제 변수 및 독립 변수 확정 (시간대 제외)
# =========================================================

# (1) 시즌 내 진행도 (season_progression: 0~1)
df_total['season_progression'] = df_total['race'].map(race_order) / 24

# (2) 독립/종속 변수 전처리
df_total['unexpected'] = df_total['unexpected'].fillna(0).astype(int)
df_total['importance'] = df_total['importance'].fillna(0).astype(int)
df_total['is_toxic'] = (df_total['Toxicity_Score'] >= 0.5).astype(int)


# =========================================================
# 3. 기술통계량 출력
# =========================================================
def print_descriptive_stats(df):
    num_cols = ['custom_interval', 'LABEL_0', 'LABEL_3', 'LABEL_1', 'LABEL_5',
                'is_toxic', 'unexpected', 'importance', 'season_progression', 'inrace_progression']

    num_stats = df[num_cols].describe().T[['count', 'mean', 'std', 'min', 'max']]

    print(f"\n{'=' * 85}")
    print(f"{'표 1. 주요 변수 기술통계량 (F1 데이터셋)':^85}")
    print(f"{'=' * 85}")
    print(num_stats.to_string(formatters={'count': '{:,.0f}'.format, 'mean': '{:,.3f}'.format}))
    print(f"{'=' * 85}")


def print_paper_format(model_res, title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")
    # 회귀 분석 결과 테이블 출력
    print(model_res.summary().tables[1])

    # 적합도 지표 계산
    if hasattr(model_res, 'rsquared'):
        r2, r2_label = model_res.rsquared, "R-squared"
        log_like = model_res.llf
    elif hasattr(model_res, 'prsquared'):
        r2, r2_label = model_res.prsquared, "Pseudo R-squared"
        log_like = model_res.llf
        # 로지스틱의 경우 모델 전체 유의성(LR p-value) 추가
        lr_p = model_res.llr_pvalue
    else:
        r2, r2_label = "N/A", "R-squared"
        log_like = "N/A"

    # 출력부
    if isinstance(r2, float):
        print(f"{r2_label}: {r2:.4f}")

    print(f"Log-Likelihood: {log_like:.3f}")

    # 로지스틱 회귀인 경우 AIC와 모델 유의성 추가 출력
    if hasattr(model_res, 'prsquared'):
        print(f"AIC: {model_res.aic:.2f}")
        print(f"LR p-value: {model_res.llr_pvalue:.4f}")

    print(f"N (표본수): {int(model_res.nobs):,}")
print_descriptive_stats(df_total)

# =========================================================
# 4. 회귀 분석 실행
# =========================================================
# 통제변수: season_progression, inrace_progression (시간대 제외)
formula_base = ' ~ unexpected + importance + season_progression + inrace_progression'

# (1) RQ1: 댓글 간격
res_interval = smf.ols('custom_interval' + formula_base, data=df_total).fit()
print_paper_format(res_interval, "표 6. 종속변수: 댓글 간격 (종목 F)")

# (2) RQ2: 4대 감정 분석
target_emotions = [('Anger', 'LABEL_0'), ('Happiness', 'LABEL_3'), ('Disgust', 'LABEL_1'), ('Surprise', 'LABEL_5')]
for name, col in target_emotions:
    res_emo = smf.ols(f'{col}' + formula_base, data=df_total).fit()
    print_paper_format(res_emo, f"종목 F {name} 감정 분석 결과")

# (3) RQ3: 독성 분석 (Logit)
res_toxic = smf.logit('is_toxic' + formula_base, data=df_total).fit()
print_paper_format(res_toxic, "종목 F 독성 분석 결과")
print("\n--- 독성 분석 Odds Ratios ---")
print(np.exp(res_toxic.params))