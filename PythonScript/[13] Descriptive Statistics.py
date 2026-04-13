import pandas as pd
import numpy as np
import glob
import os

# 1. 경로 설정
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

races = ['Bahrain', 'Australia', 'Japan', 'Imola', 'Spain', 'United_Kingdom',
         'Hungary', 'Netherlands', 'Singapore', 'Las_Vegas', 'Qatar', 'Abu_Dhabi']

all_data = []

print("🔄 경기 중(During Lap) 데이터 추출 및 간격 계산 중...")
for race in races:
    e_file = os.path.join(emo_path, f'Community_{race}_Emotion_Scores.csv')
    t_file = os.path.join(tox_path, f'Community_{race}_Toxicity_Scores.csv')

    if os.path.exists(e_file) and os.path.exists(t_file):
        df_e = pd.read_csv(e_file)
        df_t = pd.read_csv(t_file)

        # 데이터 병합 및 레이스 정보 추가
        df_race = pd.concat([df_e, df_t[['Toxicity_Score']]], axis=1)
        df_race['race'] = race

        # 🌟 수식 적용: 댓글 간 시간 간격(custom_interval) 계산
        df_race['post_timestamp'] = pd.to_datetime(df_race['post_timestamp'])
        df_race = df_race.sort_values('post_timestamp')

        diff_prev = df_race['post_timestamp'].diff().dt.total_seconds()
        diff_next = df_race['post_timestamp'].diff(-1).abs().dt.total_seconds()
        df_race['custom_interval'] = (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2

        # 🌟 [핵심] 랩타임 내에 있는 데이터만 필터링 (NaN 제외)
        df_filtered = df_race.dropna(subset=['LapNumber']).copy()
        all_data.append(df_filtered)

# 전체 통합
df_total = pd.concat(all_data, axis=0, ignore_index=True)
total_n = len(df_total)

# =========================================================
# 2. 댓글 간격 기술통계 (표 n 양식)
# =========================================================
interval_stats = {
    '항목': ['댓글 수', '평균', '표준편차', '최솟값', '중앙값', '최댓값'],
    '종목 F (경기 중)': [
        f"{total_n:,}",
        f"{df_total['custom_interval'].mean():.3f}",
        f"{df_total['custom_interval'].std():.3f}",
        f"{df_total['custom_interval'].min():.4f}",
        f"{df_total['custom_interval'].median():.4f}",
        f"{df_total['custom_interval'].max():.2f}"
    ]
}

# =========================================================
# 3. 독성 댓글 분포 (표 n 양식)
# =========================================================
toxic_threshold = 0.5
toxic_count = (df_total['Toxicity_Score'] >= toxic_threshold).sum()
toxic_ratio = (toxic_count / total_n) * 100

toxicity_stats = {
    '항목': ['독성 댓글 수', '독성 댓글 비율'],
    '종목 F (경기 중)': [
        f"{toxic_count:,}",
        f"{toxic_ratio:.1f}%"
    ]
}

# 결과 출력
print(f"\n[ 표 n. 종목별 댓글 간격 분포 (종목 F - 경기 중) ]")
print("-" * 45)
print(pd.DataFrame(interval_stats).to_string(index=False))

print(f"\n\n[ 표 n. 종목별 독성 댓글 분포 (종목 F - 경기 중) ]")
print("-" * 45)
print(pd.DataFrame(toxicity_stats).to_string(index=False))