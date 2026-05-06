import fastf1
import pandas as pd
import os
from datetime import datetime
import pytz  # 시간대 처리를 위한 라이브러리 추가

# 1. 캐시 폴더 경로 지정
CACHE_DIR = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTime/Cache'

# --- 📌 안정성 강화: 캐시 디렉토리 생성 로직 추가 ---
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"✅ 캐시 디렉토리 '{CACHE_DIR}'를 생성했습니다.")
# ----------------------------------------------------

# 1. 캐시 활성화
fastf1.Cache.enable_cache(CACHE_DIR)

# 2. 데이터 가져올 세션 정의
YEAR = 2024
GRAND_PRIX = 'Imola'
SESSION_TYPE = 'R'  # 'R'은 Race

# --- 📌 저장 경로 설정 ---
SAVE_DIRECTORY = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTime'
FILENAME_BASE = f'{YEAR}_{GRAND_PRIX.replace(" ", "_")}_Race_Lap_Times_KST.csv'  # 파일명에 KST 추가
FULL_FILE_PATH = os.path.join(SAVE_DIRECTORY, FILENAME_BASE)
# ------------------------

print(f"데이터를 로딩 중: {YEAR} {GRAND_PRIX} {SESSION_TYPE}")

# 3. 세션 데이터 로드
try:
    session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION_TYPE)
    session.load(laps=True, telemetry=False, weather=False)

except Exception as e:
    print(f"\n❌ 데이터 로딩 중 오류 발생: {e}")
    print(f"FastF1 서버에 {YEAR}년 {GRAND_PRIX} 레이스 데이터가 아직 없거나, 세션 정의에 오류가 있을 수 있습니다.")
    exit()

# 4. 랩 데이터 추출 및 정리
laps = session.laps

# KST 시간대 정의
KST = pytz.timezone('Asia/Seoul')

# 세션의 공식 시작 시간 (UTC)
session_start_utc = session.date

# --- 🛠️ 오류 해결을 위한 핵심 수정 부분 ---
# 1. session_start_utc (tz-naive)에 UTC 시간대 정보 부여 (localize)
# 2. UTC 시간대 정보가 부여된 객체를 KST로 변환 (convert)
session_start_kst = session_start_utc.tz_localize('UTC').tz_convert(KST)
# ----------------------------------------
print(f"✅ 레이스 시작 시간 (KST): {session_start_kst.strftime('%Y-%m-%d %H:%M:%S')}")

# 5. CSV로 저장 (KST 시간 변환 포함)
try:
    # ... (이하 코드는 이전과 동일)

    # 저장 디렉토리가 존재하지 않으면 생성
    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    # 데이터 정리 (IsRCL 제거)
    lap_data_to_save = laps[
        ['Driver', 'LapNumber', 'LapTime', 'LapStartTime', 'Time', 'IsPersonalBest']].copy()

    # Timedelta 객체를 초(seconds) 단위로 변환 (경과 시간)
    lap_data_to_save['LapStartTime_seconds'] = lap_data_to_save['LapStartTime'].apply(
        lambda x: x.total_seconds() if pd.notna(x) else None
    )
    lap_data_to_save['LapFinishTime_seconds'] = lap_data_to_save['Time'].apply(
        lambda x: x.total_seconds() if pd.notna(x) else None
    )

    # KST 시각으로 변환하는 새로운 컬럼 생성
    lap_data_to_save['LapStartTime_KST'] = lap_data_to_save['LapStartTime_seconds'].apply(
        lambda s: (session_start_kst + pd.Timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if pd.notna(
            s) else None
    )
    lap_data_to_save['LapFinishTime_KST'] = lap_data_to_save['LapFinishTime_seconds'].apply(
        lambda s: (session_start_kst + pd.Timedelta(seconds=s)).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] if pd.notna(
            s) else None
    )
    # ------------------------------------------------

    # 최종 저장할 데이터프레임 (timedelta 컬럼 제거)
    final_lap_data = lap_data_to_save.drop(columns=['LapStartTime', 'Time'])

    # 최종 CSV 파일 저장
    FULL_FILE_PATH = os.path.join(SAVE_DIRECTORY, FILENAME_BASE)
    final_lap_data.to_csv(FULL_FILE_PATH, index=False, encoding='utf-8')

    print("\n--- 결과 요약 ---")
    print(f"✅ 데이터 저장 성공!")
    print(f"파일 저장 경로: **{FULL_FILE_PATH}**")
    print(f"총 {len(final_lap_data)}개의 랩 기록이 저장되었습니다.")

except Exception as e:
    print(f"\n❌ 파일 저장 중 오류 발생: {e}")
    print("파일 경로 또는 쓰기 권한을 확인해 주세요.")