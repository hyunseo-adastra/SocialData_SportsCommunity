import pandas as pd
import glob
import os

# 1. 경로 설정
result_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EmotionResults'
all_files = glob.glob(os.path.join(result_path, "Community_*_Emotion_Scores.csv"))

# 2. 모든 레이스 데이터 통합 및 필터링
if not all_files:
    print("❌ 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f)
        # 🌟 [추가] 경기 진행 중에 해당하는 데이터만 필터링 (LapNumber가 있는 경우)
        if 'LapNumber' in temp_df.columns:
            temp_df = temp_df.dropna(subset=['LapNumber'])
        df_list.append(temp_df)

    df_total = pd.concat(df_list, axis=0, ignore_index=True)

    # 3. 감정 라벨 매핑
    emo_mapping = {
        'LABEL_0': '분노', 'LABEL_1': '혐오', 'LABEL_2': '공포',
        'LABEL_3': '행복', 'LABEL_4': '슬픔', 'LABEL_5': '놀람'
    }
    labels = list(emo_mapping.keys())

    # 4. 주요 감정 판별 및 점수 집계 준비
    df_total['primary_emotion'] = df_total[labels].idxmax(axis=1)
    total_comments = len(df_total)  # 필터링된 후의 전체 개수

    summary_data = []
    display_order = ['행복', '분노', '놀람', '슬픔', '혐오', '공포']
    inv_map = {v: k for k, v in emo_mapping.items()}

    # 5. 기술통계 집계 (표 5 형식)
    for emo_name in display_order:
        label_id = inv_map[emo_name]

        # A. 빈도 및 비율
        count = (df_total['primary_emotion'] == label_id).sum()
        percentage = (count / total_comments) * 100

        # B. 점수 분포 계산
        avg_score = df_total[label_id].mean()
        std_score = df_total[label_id].std()
        min_score = df_total[label_id].min()
        max_score = df_total[label_id].max()

        summary_data.append({
            '감정 라벨': emo_name,
            '빈도 (비율)': f"{count:,} ({percentage:.1f}%)",
            '평균 점수(표준편차)': f"{avg_score:.3f}({std_score:.3f})",
            '최솟값': f"{min_score:.3f}",
            '최댓값': f"{max_score:.3f}"
        })

    # 6. 결과 출력 및 저장
    final_table = pd.DataFrame(summary_data)

    print(f"\n[ 표 5. 종목 F 감정 데이터 기술통계 상세 (경기 진행 중 한정) ]")
    print(f"총 분석 데이터 수(n): {total_comments:,}")
    print("=" * 70)
    print(final_table.to_string(index=False))

    final_table.to_excel("Emotion_Stats_F_InGame_Final.xlsx", index=False)
    print(f"\n✅ 경기 중 데이터만 반영된 결과가 저장되었습니다.")