import pandas as pd
import numpy as np
import glob
import os

# 1. 경로 설정
base_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity'
event_file = os.path.join(base_path, 'EventLogData/F1 Event Data.xlsx')
emo_path = os.path.join(base_path, 'EmotionResults')
tox_path = os.path.join(base_path, 'ToxicityResults')

# [추가] 이벤트 데이터 로드 (분석 시작 전 필수 단계)
events = pd.DataFrame()
try:
    events = pd.read_excel(event_file)
    events.columns = events.columns.str.strip()  # 컬럼명 공백 제거

    # 숫자형 변환 및 전처리
    events['unexpected'] = pd.to_numeric(events['unexpected'], errors='coerce').fillna(0).astype(int)
    events['importance'] = pd.to_numeric(events['importance'], errors='coerce').fillna(0).astype(int)
    events['lap_num'] = pd.to_numeric(events['lap'], errors='coerce')

    # 병합을 위한 요약본 생성 (레이스와 랩 기준)
    event_summary = events.dropna(subset=['lap_num', 'race']).groupby(['race', 'lap_num']).agg({
        'unexpected': 'max',
        'importance': 'max'
    }).reset_index()
    print("✅ 이벤트 로그 데이터 로드 및 전처리 완료")
except Exception as e:
    print(f"❌ 이벤트 로드 오류: {e}")
    raise SystemExit("이벤트 파일이 없거나 형식이 잘못되어 분석을 중단합니다.")

races = ['Bahrain', 'Australia', 'Japan', 'Imola', 'Spain', 'United_Kingdom',
         'Hungary', 'Netherlands', 'Singapore', 'Las_Vegas', 'Qatar', 'Abu_Dhabi']

all_data = []

print("🔄 경기 중(During Lap) 데이터 추출 및 병합 중...")
for race in races:
    e_file = os.path.join(emo_path, f'Community_{race}_Emotion_Scores.csv')
    t_file = os.path.join(tox_path, f'Community_{race}_Toxicity_Scores.csv')

    if os.path.exists(e_file) and os.path.exists(t_file):
        df_e = pd.read_csv(e_file)
        df_t = pd.read_csv(t_file)

        df_race = pd.concat([df_e, df_t[['Toxicity_Score']]], axis=1)
        df_race['race'] = race

        # 간격 계산
        df_race['post_timestamp'] = pd.to_datetime(df_race['post_timestamp'])
        df_race = df_race.sort_values('post_timestamp')
        diff_prev = df_race['post_timestamp'].diff().dt.total_seconds()
        diff_next = df_race['post_timestamp'].diff(-1).abs().dt.total_seconds()
        df_race['custom_interval'] = (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2

        # 🌟 이벤트 데이터와 병합 (unexpected, importance 정보 주입)
        df_race = pd.merge(df_race, event_summary, left_on=['race', 'LapNumber'], right_on=['race', 'lap_num'],
                           how='left')

        # 경기 중 데이터만 필터링
        df_filtered = df_race.dropna(subset=['LapNumber']).copy()
        all_data.append(df_filtered)

# 전체 통합
df_total = pd.concat(all_data, axis=0, ignore_index=True)

# 결측치 채우기 (사건이 없는 랩의 데이터는 0)
df_total['unexpected'] = df_total['unexpected'].fillna(0).astype(int)
df_total['importance'] = df_total['importance'].fillna(0).astype(int)

# =========================================================
# 2. 결과 출력 (연구 방법 및 기술통계용)
# =========================================================

# 1) 사건 자체의 통계 (F1 Event Data.xlsx 기준)
total_event_f = len(events)
unexpected_event_f = (events['unexpected'] == 1).sum()
impact_event_f = (events['importance'] == 1).sum()
race_count = len(events['race'].unique())

# 2) 작성물(댓글) 매칭 통계
total_n = len(df_total)
unexp_match_n = (df_total['unexpected'] == 1).sum()
impact_match_n = (df_total['importance'] == 1).sum()

print(f"\n{'=' * 20} 논문 [연구 방법] 삽입용 수치 {'=' * 20}")
print(f"1. 종목 F 사건 통계:")
print(f"   - 총 사건 수: {total_event_f}건 (레이스 당 평균 {total_event_f / race_count:.1f}건)")
print(f"   - 의외성이 높은 사건: {unexpected_event_f}건")
print(f"   - 결과 영향성이 높은 사건: {impact_event_f}건")

print(f"\n2. 작성물(온라인 반응) 매칭 통계 (n={total_n:,}):")
print(f"   - 의외성 사건 매칭: {unexp_match_n:,}건 ({(unexp_match_n / total_n) * 100:.2f}%)")
print(f"   - 영향성 사건 매칭: {impact_match_n:,}건 ({(impact_match_n / total_n) * 100:.2f}%)")
print('=' * 60)

# (이후 간격 기술통계, 독성 분포 출력 코드는 기존과 동일하게 유지)