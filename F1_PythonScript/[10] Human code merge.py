import pandas as pd
import os

# =============================================================================
# 1. 파일 경로 설정
# =============================================================================
base_path = "/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData"

file1_path = os.path.join(base_path, "humancoding_1.csv")
file2_path = os.path.join(base_path, "humancoding_2.csv")

output_path = os.path.join(base_path, "humancoded_testset.csv")

# =============================================================================
# 2. 데이터 병합
# =============================================================================
print("데이터 병합 작업 시작...")

try:
    # 파일 로드 (한글 깨짐 방지를 위해 utf-8-sig 또는 cp949 시도)
    try:
        df1 = pd.read_csv(file1_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df1 = pd.read_csv(file1_path, encoding='cp949')
    print(f"- 첫 번째 파일 로드 완료: {len(df1)}개 행")

    try:
        df2 = pd.read_csv(file2_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        df2 = pd.read_csv(file2_path, encoding='cp949')
    print(f"- 두 번째 파일 로드 완료: {len(df2)}개 행")

    # 병합 (concat)
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # =============================================================================
    # 3. 저장
    # =============================================================================
    merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print("병합 완료!")
    print(f"총 데이터 개수: {len(merged_df)}개")
    print(f"저장 경로: {output_path}")
    print(merged_df.head())

except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
except Exception as e:
    print(f"오류 발생: {e}")