import pandas as pd
import os
import pytz

# --- 📌 사용자 정의 변수 및 경로 ---
GRAND_PRIX = 'Qatar'
YEAR = 2024
BASE_DIR = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTimeData'

# 입력 파일명
LAP_AVG_FILE_IN = os.path.join(
    BASE_DIR,
    f'{YEAR}_{GRAND_PRIX}_Lapwise_Average_Lap_Times_KST.csv'
)
# 수정된 데이터를 저장할 출력 파일명
LAP_AVG_FILE_OUT = os.path.join(
    BASE_DIR,
    f'{YEAR}_{GRAND_PRIX}_Lapwise_Average_Lap_Times_KST_Shifted.csv'
)
# ------------------------

# 1. 파일 로드
try:
    # 랩 평균 파일 로드
    df_lap = pd.read_csv(LAP_AVG_FILE_IN)
    print(f"✅ 평균 랩타임 파일 로드 성공: {os.path.basename(LAP_AVG_FILE_IN)}")
except Exception as e:
    print(f"❌ 파일 로드 오류: 경로를 다시 확인해 주세요. 오류: {e}")
    exit()

# 2. 시간 컬럼 변환 및 1시간 2분 시프트 적용
KST = pytz.timezone('Asia/Seoul')
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

# 🌟 수정된 부분: 1시간(60분) + 2분 = 총 60분을 뺌, 2분을 추가
TIME_SHIFT = pd.Timedelta(hours=-1, minutes=+9)
try:
    # 랩 시작 시각 조정
    df_lap['Avg_LapStartTime_KST_DT'] = pd.to_datetime(
        df_lap['Avg_LapStartTime_KST'], format=TIME_FORMAT
    ).dt.tz_localize(KST)
    df_lap['Avg_LapStartTime_KST'] = (
        df_lap['Avg_LapStartTime_KST_DT'] + TIME_SHIFT
    ).dt.strftime(TIME_FORMAT).str.slice(stop=-3)

    # 랩 종료 시각 조정
    df_lap['Avg_LapFinishTime_KST_DT'] = pd.to_datetime(
        df_lap['Avg_LapFinishTime_KST'], format=TIME_FORMAT
    ).dt.tz_localize(KST)
    df_lap['Avg_LapFinishTime_KST'] = (
        df_lap['Avg_LapFinishTime_KST_DT'] + TIME_SHIFT
    ).dt.strftime(TIME_FORMAT).str.slice(stop=-3)

    # 임시 컬럼 제거
    df_lap.drop(columns=['Avg_LapStartTime_KST_DT', 'Avg_LapFinishTime_KST_DT'], inplace=True)

except Exception as e:
    print(f"\n❌ 시간 변환/조정 오류: {e}")
    exit()

# 3. 새로운 CSV 파일로 저장
# 원본 파일의 컬럼 구성 유지
df_lap.to_csv(LAP_AVG_FILE_OUT, index=False, encoding='utf-8')

print("\n--- 결과 요약 ---")
print(f"✅ 평균 랩타임 데이터 1시간 2분 조정 저장 성공!")
print(f"적용된 조정: -{abs(TIME_SHIFT.total_seconds()) / 60:.0f} 분")
print(f"생성된 파일명: **{os.path.basename(LAP_AVG_FILE_OUT)}**")
print("이제 이 파일로 다시 댓글 수 계산 스크립트를 실행하세요.")