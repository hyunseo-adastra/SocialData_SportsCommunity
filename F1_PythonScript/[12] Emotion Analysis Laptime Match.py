import pandas as pd
import numpy as np
import os

# =========================================================
# 1. 파일 경로 설정
# =========================================================
# 감정 분석 결과 파일 (개별 댓글 단위)
emotion_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/AnalysisResults/f1_community_prediction_result.csv'
# 이벤트 로그 데이터
event_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/AnalysisResults/F1 Event Data.xlsx'
# 결과 저장 경로 (AnalysisResults 폴더에 저장)
output_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/AnalysisResults/f1_emotion_with_event.csv'

# =========================================================
# 2. 데이터 로드
# =========================================================
print(f"데이터 로드 중...\n - Emotion: {emotion_path}\n - Event: {event_path}")
try:
    df_emotion = pd.read_csv(emotion_path)
    df_event = pd.read_excel(event_path)
except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
    exit()


# =========================================================
# 3. 공통 키(Key) 생성 함수 (이전 로직 재사용)
# =========================================================
def make_common_lap_key(val):
    """
    어떤 입력값(1, '1', '1.0', 'Lap 1', 'Finish')이 들어와도
    통일된 포맷('Lap 1', 'After Lap')으로 변환하는 함수
    """
    s = str(val).strip().lower()

    if s == 'finish' or s == 'after lap':
        return 'After Lap'

    # 숫자만 추출 ('Lap 1' -> '1', '1.0' -> '1')
    clean_s = s.replace('lap', '').strip()

    try:
        num = int(float(clean_s))
        return f"Lap {num}"
    except:
        return str(val).strip()  # 변환 실패 시 원본 유지


# =========================================================
# 4. 매칭 키 적용
# =========================================================
print("매칭 키 생성 중...")

# (1) Race 이름 통일
if 'race' in df_emotion.columns:
    df_emotion['race'] = df_emotion['race'].astype(str).str.strip()
else:
    print("경고: Emotion 데이터에 'race' 컬럼이 없습니다.")

if 'race' in df_event.columns:
    df_event['race'] = df_event['race'].astype(str).str.strip()
else:
    # 혹시 컬럼명이 다를 경우를 대비해 확인 (예: 'Grand Prix')
    print(f"경고: Event 데이터에 'race' 컬럼이 없습니다. 컬럼 목록: {df_event.columns}")

# (2) Emotion 데이터의 Lap 컬럼 찾기 및 변환
# 데이터마다 컬럼명이 다를 수 있어 자동 탐색
lap_col = None
candidates = ['lap', 'LapNumber', 'Lap_Number', 'Lap']
for col in candidates:
    if col in df_emotion.columns:
        lap_col = col
        break

if lap_col:
    print(f"- Emotion 데이터 Lap 컬럼 감지: '{lap_col}'")
    df_emotion['join_lap'] = df_emotion[lap_col].apply(make_common_lap_key)
else:
    raise ValueError(f"Emotion 데이터에서 Lap 컬럼을 찾을 수 없습니다. 후보: {candidates}")

# (3) Event 데이터의 Lap 컬럼 찾기 및 변환
event_lap_col = 'Lap'
for col in df_event.columns:
    if 'lap' in col.lower():
        event_lap_col = col
        break
df_event['join_lap'] = df_event[event_lap_col].apply(make_common_lap_key)

# =========================================================
# 5. 병합 (Merge)
# =========================================================
print("데이터 병합 중...")

cols_to_use = ['race', 'join_lap', 'ev_unexp', 'ev_resp', 'ev_out']

# Left Join: 감정 데이터(댓글)를 기준으로 이벤트 정보를 붙임
merged_df = pd.merge(df_emotion, df_event[cols_to_use],
                     on=['race', 'join_lap'],
                     how='left')

# =========================================================
# 6. 이벤트 마커 및 후처리
# =========================================================
# 이벤트 발생 여부 마킹 (1: 발생, 0: 미발생)
merged_df['event_marker'] = np.where(
    merged_df[['ev_unexp', 'ev_resp', 'ev_out']].notna().any(axis=1),
    1, 0
)

# NaN 채우기 (이벤트가 없는 행은 0으로 처리)
merged_df['ev_unexp'] = merged_df['ev_unexp'].fillna(0)
merged_df['ev_resp'] = merged_df['ev_resp'].fillna(0)
merged_df['ev_out'] = merged_df['ev_out'].fillna(0)

# 임시 키 제거
merged_df.drop(columns=['join_lap'], inplace=True)

# =========================================================
# 7. 저장
# =========================================================
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("-" * 30)
print("작업 완료!")
print(f"총 데이터 개수: {len(merged_df)}")
print(f"이벤트 매칭된 댓글 수: {merged_df['event_marker'].sum()}")
print(f"저장 경로: {output_path}")

# 샘플 출력
if merged_df['event_marker'].sum() > 0:
    print("\n[이벤트 매칭된 데이터 예시]")
    # 결과 확인을 위해 주요 컬럼만 출력
    cols_show = ['race', lap_col, 'ev_unexp', 'ev_resp', 'ev_out']
    # pred_label이 있다면 포함
    if 'pred_label' in merged_df.columns:
        cols_show.append('pred_label')

    print(merged_df[merged_df['event_marker'] == 1][cols_show].head())
else:
    print("\n[주의] 매칭된 이벤트가 없습니다. Race 이름이나 Lap 포맷을 확인하세요.")