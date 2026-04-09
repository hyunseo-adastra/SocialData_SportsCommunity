import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# --- 📌 경로 및 변수 설정 ---
YEAR = 2024
# 루프 리스트 (파일 명칭과 일치하도록 언더바 유지 혹은 공백 사용 상관없음)
races = [
    'Bahrain', 'Australia', 'Japan', 'Imola', 'Spain',
    'United Kingdom', 'Hungary', 'Netherlands', 'Singapore',
    'Las Vegas', 'Qatar', 'Abu Dhabi'
]

BASE_DIR_COMMUNITY = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData'
ALL_EVENTS_FILE = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EventLogData/F1 Event Data.xlsx'

# 1. 통합 이벤트 로그 데이터 로드
try:
    df_all_events = pd.read_excel(ALL_EVENTS_FILE)

    # [핵심] 비교를 위해 엑셀 내의 모든 레이스 이름을 '공백' 기준으로 통일합니다.
    df_all_events['race'] = df_all_events['race'].str.replace('_', ' ')

    print(f"✅ 통합 엑셀 파일 로드 성공: {len(df_all_events)}개의 이벤트 발견")
except Exception as e:
    print(f"❌ 엑셀 파일 로드 실패: {e}")
    exit()

# --- 🔄 모든 레이스 루프 시작 ---
for RACE_NAME in races:
    # 2. 이름 표준화 (공백 기준)
    # 파일명에는 언더바가 쓰일 수 있으므로 구분을 둡니다.
    pure_name = RACE_NAME.replace('_', ' ')  # 'Abu Dhabi'
    file_name_part = RACE_NAME.replace(' ', '_')  # 'Abu_Dhabi'

    print(f"\n🚀 {pure_name} 분석 시작...")

    # 경로 설정
    COMMENT_COUNT_FILE = os.path.join(BASE_DIR_COMMUNITY, f'{file_name_part}_Lapwise_Comment_Count.csv')
    PLOT_FILE = f'{YEAR}_{file_name_part}_Event_Reaction_Analysis.png'

    try:
        # 3. 레이스별 댓글 데이터 로드
        if not os.path.exists(COMMENT_COUNT_FILE):
            print(f"⚠️ {file_name_part} 댓글 파일이 없습니다. 경로 확인: {COMMENT_COUNT_FILE}")
            continue

        df_comments = pd.read_csv(COMMENT_COUNT_FILE)

        # 4. 데이터 정리: 댓글 데이터
        df_comments['LapNumber'] = pd.to_numeric(df_comments['Lap_Label'], errors='coerce')
        df_comments_race = df_comments.dropna(subset=['LapNumber']).copy()
        df_comments_race['LapNumber'] = df_comments_race['LapNumber'].astype(int)

        # 5. [핵심] 통합 로그에서 해당 레이스 필터링 (둘 다 공백 기준이므로 매칭 성공)
        df_events = df_all_events[df_all_events['race'] == pure_name].copy()

        if df_events.empty:
            print(f"ℹ️ {pure_name}에 대한 이벤트 로그를 찾지 못했습니다.")
        else:
            print(f"✅ {pure_name} 이벤트 {len(df_events)}건 매칭 완료.")

        # 6. 컬럼 표준화 및 타입 변환
        if 'lap' in df_events.columns:
            df_events.rename(columns={'lap': 'LapNumber'}, inplace=True)
        if 'domain' in df_events.columns:
            df_events.rename(columns={'domain': 'Category'}, inplace=True)

        df_events['LapNumber'] = pd.to_numeric(df_events['LapNumber'], errors='coerce').fillna(0).astype(int)
        df_events['unexpectedness'] = pd.to_numeric(df_events['unexpected'], errors='coerce').fillna(0).astype(int)
        df_events['outcome_relevance'] = pd.to_numeric(df_events['importance'], errors='coerce').fillna(0).astype(int)

        df_events_unique = df_events[['LapNumber', 'Category', 'unexpectedness', 'outcome_relevance']].drop_duplicates()

        # 7. 데이터 통합
        df_analysis = pd.merge(df_comments_race, df_events_unique, on='LapNumber', how='left')
        df_analysis['Has_Event'] = df_analysis['Category'].notna().astype(int)
        df_analysis['Major_Event'] = (
                    (df_analysis['unexpectedness'] == 1) | (df_analysis['outcome_relevance'] == 1)).astype(int)

        # 8. 시각화
        plt.figure(figsize=(16, 7))
        plt.plot(df_analysis['LapNumber'], df_analysis['Comment_Count'], label='Comment Count', color='gray',
                 linestyle='-', alpha=0.6)

        event_laps = df_analysis[df_analysis['Has_Event'] == 1]
        major = event_laps[event_laps['Major_Event'] == 1]
        plt.scatter(major['LapNumber'], major['Comment_Count'], c='red', marker='o', s=80, label='Major Event',
                    zorder=5)

        minor = event_laps[event_laps['Major_Event'] == 0]
        plt.scatter(minor['LapNumber'], minor['Comment_Count'], c='blue', marker='o', s=50, label='Minor Event',
                    zorder=4)

        plt.xlabel("Lap Number", fontsize=12)
        plt.ylabel("Comment Count", fontsize=12)
        plt.title(f"{pure_name} {YEAR} - Comment Count by Event", fontsize=14)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # 9. 저장 및 닫기
        plt.savefig(PLOT_FILE)
        plt.close()
        print(f"✅ 저장 성공: {PLOT_FILE}")

    except Exception as e:
        print(f"❌ {pure_name} 처리 중 오류: {e}")

print("\n✨ 모든 분석이 성공적으로 끝났습니다!")