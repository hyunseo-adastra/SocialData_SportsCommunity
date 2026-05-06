import pandas as pd
import numpy as np

df = pd.read_csv('/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData/f1_community_total.csv')

# =========================================================
# 전처리: 계산을 위한 준비
# =========================================================

# 1) 타임스탬프 변환 (문자열 -> datetime)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2) 댓글 길이(글자수) 계산
# NaN인 경우 0으로 처리 후 길이 잼
df['text_len'] = df['Text'].fillna("").astype(str).apply(len)

# 3) 정렬 (★매우 중요★)
# 시간 차이를 계산하려면 반드시 시간순으로 정렬되어 있어야 합니다.
df = df.sort_values(by=['race', 'lap', 'timestamp'])

# =========================================================
# 핵심 로직: 댓글 간 시간 차이(Time Gap) 계산
# =========================================================

# 같은 경기(race), 같은 랩(Lap_Number) 안에서 '현재 댓글 시간 - 이전 댓글 시간'을 구합니다.
# transform을 쓰면 집계된 값을 원래 데이터프레임의 길이에 맞게 쫙 뿌려줍니다.
df['time_gap'] = df.groupby(['race', 'lap'])['timestamp'].diff()

# time_gap은 '0 days 00:00:12' 같은 timedelta 형태이므로, 이를 '초(Seconds)' 단위 숫자로 바꿉니다.
df['time_gap_sec'] = df['time_gap'].dt.total_seconds()

# =========================================================
# 최종 집계: Race & Lap 별 그룹화
# =========================================================

# Lap_Number가 NaN인 데이터(랩 정보 없음)는 제외하고 볼까요? (선택사항)
df_analysis = df.dropna(subset=['lap'])

grouped_stats = df_analysis.groupby(['race', 'lap']).agg(

    # 1. 댓글 수 (화력)
    comment_count=('Text', 'count'),

    # 2. 평균 댓글 간격 (초 단위) - 낮을수록 긴박함
    avg_time_gap=('time_gap_sec', 'mean'),

    # 3. 평균 댓글 길이 - 길수록 진지한 분석글일 확률 높음
    avg_text_len=('text_len', 'mean')

).reset_index()

# 보기 좋게 정렬 (경기 이름순 -> 랩 순서)
grouped_stats = grouped_stats.sort_values(by=['race', 'lap'])

# 결과 확인
print(grouped_stats.head(10))

# 엑셀로 저장해서 보고 싶다면
grouped_stats.to_excel("f1_race_lap_analysis.xlsx", index=False)