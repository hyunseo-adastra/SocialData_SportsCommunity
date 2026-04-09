import pandas as pd
import numpy as np
import os
import pytz

# --- 📌 사용자 정의 변수 및 경로 설정 ---
YEAR = 2024
races = [
    'Bahrain', 'Australia', 'Japan', 'Imola', 'Spain',
    'United_Kingdom', 'Hungary', 'Netherlands', 'Singapore',
    'Las_Vegas', 'Qatar', 'Abu_Dhabi'
]

# 경로 설정
BASE_DIR_LAP = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTimeData'
BASE_DIR_COMMUNITY = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData'

# 시간대 및 포맷 정의
KST = pytz.timezone('Asia/Seoul')
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

# --- 🔄 모든 레이스 루프 시작 ---
for GRAND_PRIX in races:
    print(f"\n🚀 {GRAND_PRIX} 데이터 처리 시작...")

    # 입력 파일 경로
    LAP_AVG_FILE = os.path.join(BASE_DIR_LAP, f'{YEAR}_{GRAND_PRIX}_Lapwise_Average_Lap_Times_KST_Shifted.csv')
    COMMUNITY_FILE = os.path.join(BASE_DIR_COMMUNITY, f'Community_{GRAND_PRIX}.csv')

    # 출력 파일 경로
    OUTPUT_FILE = os.path.join(BASE_DIR_COMMUNITY, f'Community_{GRAND_PRIX}_With_LapNumber.csv')

    try:
        # 1. 파일 존재 여부 확인 및 로드
        if not os.path.exists(LAP_AVG_FILE) or not os.path.exists(COMMUNITY_FILE):
            print(f"⚠️ {GRAND_PRIX}: 필요한 파일이 없어 건너뜁니다.")
            continue

        df_lap = pd.read_csv(LAP_AVG_FILE)
        df_community = pd.read_csv(COMMUNITY_FILE)

        # 2. 시간 컬럼 변환
        # 랩 데이터 시간 변환
        df_lap['Avg_LapStartTime_KST_DT'] = pd.to_datetime(df_lap['Avg_LapStartTime_KST'],
                                                           format=TIME_FORMAT).dt.tz_localize(KST)
        df_lap['Avg_LapFinishTime_KST_DT'] = pd.to_datetime(df_lap['Avg_LapFinishTime_KST'],
                                                            format=TIME_FORMAT).dt.tz_localize(KST)

        # 커뮤니티 데이터 시간 변환
        community_time_naive = pd.to_datetime(df_community['post_timestamp'], errors='coerce')
        df_community['Comment_Time_DT'] = community_time_naive.dt.tz_localize(KST)

        # 유효하지 않은 시간 데이터 제거
        df_community.dropna(subset=['Comment_Time_DT'], inplace=True)

        # 3. LapTime 구간(Interval) 설정
        lap_start_times = df_lap['Avg_LapStartTime_KST_DT'].tolist()
        last_lap_finish_time = df_lap['Avg_LapFinishTime_KST_DT'].iloc[-1]
        base_borders = lap_start_times + [last_lap_finish_time]

        # Before/After 처리를 위한 여유 구간 설정 (각 1시간)
        min_border = lap_start_times[0] - pd.Timedelta(hours=1)
        max_border = last_lap_finish_time + pd.Timedelta(hours=1)
        final_borders = [min_border] + base_borders + [max_border]

        # 레이블 설정
        lap_labels = ['Before Lap'] + df_lap['LapNumber'].astype(int).astype(str).tolist() + ['After Lap']

        # 4. Pandas.cut을 이용한 랩 할당
        df_community['Lap_Label'] = pd.cut(
            df_community['Comment_Time_DT'],
            bins=final_borders,
            labels=lap_labels,
            include_lowest=True,
            right=False
        )

        # 5. LapNumber 컬럼 추가 (Before/After는 NaN이 됨)
        df_community['LapNumber'] = pd.to_numeric(df_community['Lap_Label'], errors='coerce')

        # 6. 결과 저장
        # 분석용 임시 컬럼 제거 후 저장
        df_community_save = df_community.drop(columns=['Comment_Time_DT']).copy()
        df_community_save.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

        print(f"✅ 완료: {os.path.basename(OUTPUT_FILE)} 저장 성공!")

    except Exception as e:
        print(f"❌ {GRAND_PRIX} 처리 중 오류 발생: {e}")

print("\n✨ 모든 레이스의 랩 번호 할당 작업이 완료되었습니다!")