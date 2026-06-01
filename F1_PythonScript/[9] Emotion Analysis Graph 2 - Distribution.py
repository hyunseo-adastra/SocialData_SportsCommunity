import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 데이터 로드
# =========================
csv_path = "/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EmotionAnalysis/F1_emotion_predictions_kcelectra.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# =========================
# 2. Bahrain 필터
# =========================
df_bahrain = df[df["gp_name"] == "Bahrain"].copy()
df_bahrain["LapNumber"] = pd.to_numeric(df_bahrain["LapNumber"], errors="coerce")
df_bahrain = df_bahrain.dropna(subset=["LapNumber"])
df_bahrain["LapNumber"] = df_bahrain["LapNumber"].astype(int)
df_bahrain["emo_conf"] = pd.to_numeric(df_bahrain["emo_conf"], errors="coerce")

# =========================
# 3. 감정 최종 레이블 (confidence ≤ 0.5 → neutral)
# =========================
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

def final_label(row):
    conf = row["emo_conf"]
    emo = str(row["emo_name"]).strip().lower()

    if pd.isna(conf) or conf <= 0.5:
        return "neutral"
    return emo if emo in EMOTIONS else "neutral"

df_bahrain["emo_final"] = df_bahrain.apply(final_label, axis=1)

# =========================
# 4. Lap × Emotion 카운트 테이블
# =========================
counts = (
    df_bahrain
    .pivot_table(index="LapNumber", columns="emo_final", aggfunc="size", fill_value=0)
    .reindex(columns=EMOTIONS, fill_value=0)
    .sort_index()
)

# =========================
# 5. 카운트를 비율(0~1)로 정규화
# =========================
row_sums = counts.sum(axis=1).replace(0, np.nan)  # 0으로 나누기 방지
props = counts.div(row_sums, axis=0).fillna(0)    # 비율표

# =========================
# 6. 감정별 색상 지정
# =========================
color_map = {
    "neutral":  "gray",
    "anger":    "red",
    "happiness":"yellow",
    "sadness":  "blue",
    "surprise": "brown",
    "fear":     "purple",
    "disgust":  "green",
}

# =========================
# 7. 비율 스택형 바 플롯
# =========================
plt.figure(figsize=(16, 6))

x = props.index.values  # LapNumber
bottom = np.zeros(len(props))

for emo in EMOTIONS:
    heights = props[emo].values
    plt.bar(
        x,
        heights,
        bottom=bottom,
        color=color_map[emo],
        label=emo,
    )
    bottom += heights

plt.xlabel("Lap Number")
plt.ylabel("Proportion of comments")
plt.title("Emotion proportion per lap – Bahrain (KcELECTRA, neutral if conf ≤ 0.5)")
plt.ylim(0, 1)
plt.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()