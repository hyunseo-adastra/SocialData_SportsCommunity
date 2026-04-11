import pandas as pd
import glob
import os

# 1. 독성 분석 결과(.csv)가 저장된 폴더 경로 설정
# 본인의 실제 경로로 수정하세요.
result_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/ToxicityResults'
all_files = glob.glob(os.path.join(result_path, "Community_*_Toxicity_Scores.csv"))

# 2. 모든 레이스 데이터 통합
if not all_files:
    print("❌ 독성 분석 결과 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    li = []
    for filename in all_files:
        df_temp = pd.read_csv(filename)
        li.append(df_temp)

    df_total = pd.concat(li, axis=0, ignore_index=True)

    # 3. 독성 댓글 판별 (연구 설계에 따라 0.5 이상을 독성으로 간주)
    threshold = 0.5
    total_comments = len(df_total)
    toxic_comments_df = df_total[df_total['Toxicity_Score'] >= threshold]

    toxic_count = len(toxic_comments_df)
    toxic_ratio = (toxic_count / total_comments) * 100

    # 4. 최종 결과 출력 (표 양식)
    print(f"\n[ 표 n. 종목별 독성 댓글 분포 (종목 F 통합) ]")
    print("-" * 40)

    summary_table = pd.DataFrame({
        '항목': ['독성 댓글 수', '독성 댓글 비율'],
        '종목 F': [f"{toxic_count:,}", f"{toxic_ratio:.1f}%"]
    })

    print(summary_table.to_string(index=False))

    # 5. 엑셀 저장
    summary_table.to_excel("Table_N_Toxicity_Summary_Race_F.xlsx", index=False)
    print(f"\n✅ 요약 결과가 저장되었습니다: Table_N_Toxicity_Summary_Race_F.xlsx")