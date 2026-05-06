import pandas as pd
import numpy as np
import os
import pytz

# --- 📌 Configuration ---
races = [
    'Bahrain', 'Australia', 'Japan', 'Imola', 'Spain',
    'United_Kingdom', 'Hungary', 'Netherlands', 'Singapore',
    'Las_Vegas', 'Qatar', 'Abu_Dhabi'
]

YEAR = 2024
KST = pytz.timezone('Asia/Seoul')
TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

# --- 🏎️ Loop starts here ---
for GRAND_PRIX in races:
    print(f"\n🚀 Starting processing for: {GRAND_PRIX}...")

    # Paths updated dynamically per race
    LAP_AVG_FILE = f'/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/GlobalLapTimeData/{YEAR}_{GRAND_PRIX}_Lapwise_Average_Lap_Times_KST_Shifted.csv'
    COMMUNITY_FILE = f'/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData/Community_{GRAND_PRIX}.csv'
    OUTPUT_FILE = f'/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/CommunityData/{GRAND_PRIX}_Lapwise_Comment_Count.csv'

    try:
        # 1. Load Files
        df_lap = pd.read_csv(LAP_AVG_FILE)
        df_community = pd.read_csv(COMMUNITY_FILE)

        # 2. Time Conversion
        df_lap['Avg_LapStartTime_KST_DT'] = pd.to_datetime(df_lap['Avg_LapStartTime_KST'], format=TIME_FORMAT).dt.tz_localize(KST)
        df_lap['Avg_LapFinishTime_KST_DT'] = pd.to_datetime(df_lap['Avg_LapFinishTime_KST'], format=TIME_FORMAT).dt.tz_localize(KST)
        df_community['Comment_Time_DT'] = pd.to_datetime(df_community['post_timestamp']).dt.tz_localize(KST)

        # 3. Define Intervals
        lap_start_times = df_lap['Avg_LapStartTime_KST_DT'].tolist()
        last_lap_finish_time = df_lap['Avg_LapFinishTime_KST_DT'].iloc[-1]
        base_borders = lap_start_times + [last_lap_finish_time]

        min_border = lap_start_times[0] - pd.Timedelta(hours=1)
        max_border = last_lap_finish_time + pd.Timedelta(hours=1)
        final_borders = [min_border] + base_borders + [max_border]

        lap_labels = ['Before Lap'] + df_lap['LapNumber'].astype(str).tolist() + ['After Lap']

        # 4. Binning with pd.cut
        df_community['Lap_Label'] = pd.cut(
            df_community['Comment_Time_DT'],
            bins=final_borders,
            labels=lap_labels,
            include_lowest=True,
            right=False
        )

        # 5. Grouping
        comment_counts = df_community.groupby('Lap_Label')['Comment_Time_DT'].count().reset_index()
        comment_counts.rename(columns={'Comment_Time_DT': 'Comment_Count'}, inplace=True)

        # 6. Save Result
        comment_counts.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
        print(f"✅ Success! Saved to: {os.path.basename(OUTPUT_FILE)}")

    except FileNotFoundError:
        print(f"⚠️ Skipping {GRAND_PRIX}: File not found.")
    except Exception as e:
        print(f"❌ Error processing {GRAND_PRIX}: {e}")

print("\n✨ All races have been processed!")