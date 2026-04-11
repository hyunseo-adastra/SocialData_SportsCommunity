import pandas as pd
import numpy as np
import os

# 1. 파일 로드 (사용자 환경의 절대 경로)
file_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData/f1_community_total.csv'

if not os.path.exists(file_path):
    print(f"❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
else:
    df = pd.read_csv(file_path)

    # 2. 전처리
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # 레이스, 랩, 시간순으로 정렬
    df = df.sort_values(by=['race', 'lap', 'timestamp']).reset_index(drop=True)


    # 3. 연구 방법론 수식 적용
    def calculate_custom_interval(group):
        if len(group) <= 1:
            return pd.Series([0.0] * len(group), index=group.index)

        # tn - tn-1 (이전 댓글과의 차이)
        diff_prev = group['timestamp'].diff().dt.total_seconds()
        # tn+1 - tn (이후 댓글과의 차이)
        diff_next = group['timestamp'].diff(-1).abs().dt.total_seconds()

        # 수식 적용 및 예외 처리 (첫 댓글은 후속과의 차이, 마지막 댓글은 직전과의 차이)
        interval = (diff_prev.fillna(diff_next) + diff_next.fillna(diff_prev)) / 2
        return interval


    print("🔄 댓글 간 시간 간격 계산 중...")
    df['custom_interval'] = df.groupby(['race', 'lap'], group_keys=False).apply(calculate_custom_interval)


    # 4. 전체 데이터를 '종목 F'로 집계
    def get_final_summary(data):
        stats = data['custom_interval'].describe(percentiles=[.5])

        # 모든 수치를 소수점 3자리까지 문자열로 포맷팅 (최대, 최소 포함)
        summary_data = {
            '종목 F': [
                f"{len(data):,}",  # 댓글 수
                f"{stats['mean']:.5f}",  # 평균
                f"{stats['std']:.5f}",  # 표준편차
                f"{stats['min']:.5f}",  # 최솟값 (3자리)
                f"{stats['50%']:.5f}",  # 중앙값
                f"{stats['max']:.5f}"  # 최댓값 (3자리)
            ]
        }

        return pd.DataFrame(summary_data, index=['댓글 수', '평균', '표준편차', '최솟값', '중앙값', '최댓값'])


    # 통계표 생성
    final_table = get_final_summary(df)

    # 결과 출력
    print("\n[ 연구 질문 1: 온라인 반응의 양적 밀도 기술통계 ]")
    print(final_table)

    # 5. 결과 저장 (Excel)
    output_name = "f1_total_density_analysis_v2.xlsx"
    final_table.to_excel(output_name)
    print(f"\n✅ 분석 결과가 저장되었습니다: {output_name}")