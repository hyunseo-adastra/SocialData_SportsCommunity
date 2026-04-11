import pandas as pd
import glob
import os

# 1. 경로 설정 (사용자 환경의 실제 경로로 수정하세요)
result_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EmotionResults'
all_files = glob.glob(os.path.join(result_path, "Community_*_Emotion_Scores.csv"))

# 2. 모든 레이스 데이터 통합
if not all_files:
    print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    df_list = []
    for f in all_files:
        df_list.append(pd.read_csv(f))
    df_total = pd.concat(df_list, axis=0, ignore_index=True)

    # 3. 감정 라벨 매핑 (KOTE/KcELECTRA 모델 기준 - 본인의 모델 라벨을 확인하세요)
    # 아래는 일반적인 6감정 분류 모델의 예시 매핑입니다.
    emo_mapping = {
        'LABEL_0': '분노',
        'LABEL_1': '혐오',
        'LABEL_2': '공포',
        'LABEL_3': '행복',
        'LABEL_4': '슬픔',
        'LABEL_5': '놀람'
    }
    labels = list(emo_mapping.keys())

    # 4. 개별 댓글의 '주요 감정' 판별 (6개 확률 중 최대값인 컬럼 찾기)
    df_total['primary_emotion'] = df_total[labels].idxmax(axis=1)

    # 5. 기술통계 집계
    total_comments = len(df_total)
    summary_data = []

    # 표에 표시할 순서 정의
    display_order = ['행복', '분노', '놀람', '슬픔', '혐오', '공포']
    inv_map = {v: k for k, v in emo_mapping.items()}

    for emo_name in display_order:
        label_id = inv_map[emo_name]

        # A. 빈도 및 비율 계산 (해당 감정이 주요 감정으로 꼽힌 횟수)
        count = (df_total['primary_emotion'] == label_id).sum()
        percentage = (count / total_comments) * 100

        # B. 평균 점수 계산 (해당 감정 컬럼의 전체 평균 확률값)
        avg_score = df_total[label_id].mean()

        summary_data.append({
            '주요 감정': emo_name,
            '빈도 (비율)': f"{count:,} ({percentage:.2f}%)",
            '평균 점수': f"{avg_score:.3f}"
        })

    # 6. 결과 출력 및 저장
    final_table = pd.DataFrame(summary_data)

    print(f"\n[ 표 4. 종목 F (전체) 기술통계 결과 ]")
    print(f"총 분석 댓글 수: {total_comments:,}건")
    print("-" * 50)
    print(final_table.to_string(index=False))

    # 엑셀 파일로 저장하여 논문 작성에 활용
    final_table.to_excel("Table4_Race_F_Total_Summary.xlsx", index=False)
    print(f"\n✅ 요약 표가 저장되었습니다: Table4_Race_F_Total_Summary.xlsx")