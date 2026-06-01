import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. 파일 목록 및 경로 정의
RACE_NAMES = [
    'Bahrain', 'Australia', 'Japan', 'Imola', 'Spain', 'United_Kingdom',
    'Hungary', 'Netherlands', 'Singapore', 'Las_Vegas', 'Qatar', 'Abu_Dhabi'
]
BASE_PATH = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData'

all_data = []

# 2. 모든 파일 로드 및 데이터 통합
for race_name in RACE_NAMES:
    file_name = f'{race_name}_Lapwise_Comment_Count.csv'
    full_path = os.path.join(BASE_PATH, file_name)

    try:
        df = pd.read_csv(full_path)
    except FileNotFoundError:
        print(f"❌ 오류: 파일 '{file_name}'을 찾을 수 없습니다. (로드 건너뜀)")
        continue

    df['Race_Name'] = race_name

    # Lap_Order 생성: Before Lap=0, Lap N=N, After Lap=MaxLap+1
    df['Lap_Order_Numeric'] = pd.to_numeric(df['Lap_Label'], errors='coerce')
    max_lap = df['Lap_Order_Numeric'].max() if not df['Lap_Order_Numeric'].empty else 0


    def map_lap_order(row):
        if row['Lap_Label'] == 'Before Lap':
            return 0
        elif row['Lap_Label'] == 'After Lap':
            return max_lap + 1
        else:
            return row['Lap_Order_Numeric']


    df['Lap_Order'] = df.apply(map_lap_order, axis=1)
    all_data.append(df)

if not all_data:
    print("❌ 오류: 유효하게 로드된 파일이 없어 시각화를 진행할 수 없습니다.")
    exit()

df_combined = pd.concat(all_data, ignore_index=True)

# 3. 🌟 필터링: 'Before Lap' 및 'After Lap' 제외
df_filtered = df_combined[
    (df_combined['Lap_Label'] != 'Before Lap') &
    (df_combined['Lap_Label'] != 'After Lap')
    ].copy()

# 4. 시각화 준비: X축 틱 레이블 생성 (필터링된 데이터 사용)
tick_data = df_filtered[['Lap_Order', 'Lap_Label']].drop_duplicates().sort_values('Lap_Order')
x_ticks = tick_data['Lap_Order'].values
x_labels = tick_data['Lap_Label'].values

# 5. 시각화 (Matplotlib)
plt.figure(figsize=(18, 9))

# 각 레이스별로 선 그래프 그리기
for race in RACE_NAMES:
    df_race = df_filtered[df_filtered['Race_Name'] == race]
    if not df_race.empty:
        plt.plot(
            df_race['Lap_Order'],
            df_race['Comment_Count'],
            label=race,
            marker='o',
            markersize=3,
            linewidth=1.5
        )

# 6. 플롯 꾸미기
# 틱 간격 조정 (랩 수가 많으므로 5개 간격으로 표시)
tick_interval = 5
plt.xticks(x_ticks[::tick_interval], x_labels[::tick_interval], rotation=45, ha='right', fontsize=10)

plt.xlabel("Lap Number", fontsize=14)
plt.ylabel("Comment Count", fontsize=14)
plt.title("Lap-wise Comment Count Comparison", fontsize=16)
plt.legend(title="Race", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(rect=[0, 0, 0.85, 1])

# 7. 파일 저장
plot_file = '12_Races_Lapwise_Comment_Count_RACE_ONLY.png'
plt.savefig(plot_file)

print(f"\n✅ 시각화 파일 저장 성공: {plot_file}")
print("생성된 이미지 파일을 확인해 주세요. 오직 레이스 랩(Lap 1 ~ Final Lap)만 표시됩니다.")