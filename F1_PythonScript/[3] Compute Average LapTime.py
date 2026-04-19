import pandas as pd
import numpy as np
import os
import pytz

# --- 📌 사용자 정의 변수 ---
# 이 변수만 변경하여 다른 레이스를 처리하세요.
GRAND_PRIX = 'Abu Dhabi'
YEAR = 2024
# ------------------------

# 파일 이름 정의
file_name = f'/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTime/{YEAR}_{GRAND_PRIX.replace(" ", "_")}_Race_Lap_Times_KST.csv'
# 🌟 출력 파일명 변경: Lapwise (랩별) 평균임을 명시
output_file_name = f'/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTime/{YEAR}_{GRAND_PRIX.replace(" ", "_")}_Lapwise_Average_Lap_Times_KST.csv'

# 1. CSV 파일 로드
try:
    df = pd.read_csv(file_name)
    print(f"✅ 파일 로드 성공: {os.path.basename(file_name)}")
except FileNotFoundError:
    print(f"❌ 오류: 파일 '{file_name}'을 찾을 수 없습니다.")
    exit()

# 2. KST time 컬럼을 시간대 인식(Timezone-aware) datetime 객체로 변환
KST = pytz.timezone('Asia/Seoul')

try:
    # 문자열을 datetime 객체로 변환
    start_time_naive = pd.to_datetime(df['LapStartTime_KST'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
    finish_time_naive = pd.to_datetime(df['LapFinishTime_KST'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

    # tz_localize()로 KST 시간대 정보 부여
    df['LapStartTime_KST'] = start_time_naive.dt.tz_localize(KST)
    df['LapFinishTime_KST'] = finish_time_naive.dt.tz_localize(KST)

except Exception as e:
    print(f"❌ 시간 변환 오류: {e}")
    exit()

# 3. 데이터가 누락된 랩 데이터 필터링
# LapStartTime 또는 LapFinishTime이 NaT인 행은 제외합니다.
df_filtered = df.dropna(subset=['LapStartTime_KST', 'LapFinishTime_KST']).copy()


# 4. 🌟 LapNumber별 평균 Lap Start Time과 Finish Time 계산
average_times_df = df_filtered.groupby('LapNumber').agg(
    Avg_LapStartTime_KST=('LapStartTime_KST', 'mean'),
    Avg_LapFinishTime_KST=('LapFinishTime_KST', 'mean'),
    Num_Drivers=('Driver', 'count') # 해당 랩을 완료한 드라이버 수 확인용
).reset_index()


# 5. 결과 포맷팅 (KST 시각 문자열)
OUTPUT_TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

average_times_df['Avg_LapStartTime_KST'] = average_times_df['Avg_LapStartTime_KST'].dt.strftime(OUTPUT_TIME_FORMAT).str.slice(stop=-3)
average_times_df['Avg_LapFinishTime_KST'] = average_times_df['Avg_LapFinishTime_KST'].dt.strftime(OUTPUT_TIME_FORMAT).str.slice(stop=-3)


# 6. 새로운 CSV 파일로 저장
try:
    average_times_df.to_csv(output_file_name, index=False, encoding='utf-8')

    print("\n--- 결과 요약 ---")
    print(f"✅ 랩별 평균 시각 데이터 저장 성공!")
    print(f"생성된 파일명: **{os.path.basename(output_file_name)}**")
    print("저장된 데이터 미리보기:")
    print(average_times_df.head())
except Exception as e:
    print(f"\n❌ 최종 파일 저장 중 오류 발생: {e}")