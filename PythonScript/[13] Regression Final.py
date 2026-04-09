import os
import numpy as np
import pandas as pd
from statsmodels.formula.api import logit

# =========================================================
# 1. Load
# =========================================================
csv_path = "/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/EmotionAnalysis/f1_community_prediction_result.csv"
out_path = os.path.join(os.path.dirname(csv_path), "f1_comment_level_logit_report.txt")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print(f"Loaded: {csv_path}")
print(f"Rows: {len(df):,}")
print("Columns:", list(df.columns))

# =========================================================
# 2. Required columns (adjust if your names differ)
# =========================================================
# text column guess
TEXT_COL_CANDIDATES = ["text", "comment", "content", "body"]
text_col = next((c for c in TEXT_COL_CANDIDATES if c in df.columns), None)
if text_col is None:
    raise ValueError(f"Cannot find text column. Tried: {TEXT_COL_CANDIDATES}")

# timestamp column guess
TS_COL_CANDIDATES = ["timestamp", "time", "created_at", "datetime", "date"]
ts_col = next((c for c in TS_COL_CANDIDATES if c in df.columns), None)
if ts_col is None:
    raise ValueError(f"Cannot find timestamp column. Tried: {TS_COL_CANDIDATES}")

# event dummies must exist (or you can create from event_type later)
for col in ["ev_unexp", "ev_resp", "ev_out"]:
    if col not in df.columns:
        raise ValueError(f"Missing required event column: {col} (need it for regression)")

# =========================================================
# 3. Preprocess
# =========================================================
df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=False)
df = df.dropna(subset=[ts_col, text_col])

# Basic controls
df["comment_length"] = df[text_col].astype(str).str.len()
df["hour"] = df[ts_col].dt.hour
df["dow"] = df[ts_col].dt.dayofweek  # 0=Mon

# If event columns are not strictly 0/1, coerce them
for col in ["ev_unexp", "ev_resp", "ev_out"]:
    df[col] = (df[col].fillna(0).astype(float) > 0).astype(int)

# =========================================================
# 4. Build binary emotion targets
#    Case A: already has emotion one-hot columns (anger/happiness/...)
#    Case B: has a single label column like emotion_target or emotion
# =========================================================
EMO_ONEHOT = ["anger", "happiness", "surprise", "sadness", "fear", "disgust"]

onehot_available = [c for c in EMO_ONEHOT if c in df.columns]

if len(onehot_available) >= 2:
    # Use existing one-hot columns
    targets = onehot_available
    # Ensure 0/1
    for t in targets:
        df[t] = (df[t].fillna(0).astype(float) > 0).astype(int)

else:
    # Try label column
    LABEL_COL_CANDIDATES = ["emotion_target", "emotion", "pred_emotion", "label"]
    label_col = next((c for c in LABEL_COL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(
            "No one-hot emotion columns found, and no label column found. "
            f"Tried labels: {LABEL_COL_CANDIDATES}"
        )

    # Normalize labels to string
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # Map common label variants -> canonical names
    label_map = {
        "angry": "anger",
        "anger": "anger",
        "happy": "happiness",
        "happiness": "happiness",
        "joy": "happiness",
        "surprise": "surprise",
        "sad": "sadness",
        "sadness": "sadness",
        "fear": "fear",
        "disgust": "disgust",
    }
    df["_emo"] = df[label_col].map(label_map)

    targets = []
    for emo in EMO_ONEHOT:
        col = f"is_{emo}"
        df[col] = (df["_emo"] == emo).astype(int)
        targets.append(col)

print("Binary targets:", targets)

# =========================================================
# 5. Run Logit per emotion
# =========================================================
results = []

def run_logit(target_col: str, title: str):
    # Drop NA and check variation (need both 0 and 1)
    d = df.dropna(subset=[target_col, "ev_unexp", "ev_resp", "ev_out", "comment_length"])
    if d[target_col].nunique() < 2:
        print(f"[Skip] {target_col}: no variation (all same)")
        return

    # Model: emotion ~ event dummies + controls
    formula = f"{target_col} ~ ev_unexp + ev_resp + ev_out + comment_length + C(hour) + C(dow)"

    model = logit(formula, data=d).fit(disp=False)

    # Console summary (key terms)
    print(f"\n--- {title} ---")
    print(f"N={len(d):,} | McFadden R²={model.prsquared:.5f}")
    for term in ["ev_unexp", "ev_resp", "ev_out"]:
        coef = model.params.get(term, np.nan)
        pval = model.pvalues.get(term, np.nan)
        oratio = float(np.exp(coef)) if np.isfinite(coef) else np.nan
        star = "*" if (np.isfinite(pval) and pval < 0.05) else ""
        print(f" - {term}: coef={coef:.4f}, OR={oratio:.3f}, p={pval:.4g} {star}")

    # Save full table
    block = []
    block.append("\n" + "=" * 80)
    block.append(f"[Logit] {title}")
    block.append(f"Formula: {formula}")
    block.append("=" * 80)
    block.append(model.summary().as_text())
    block.append("\n")
    results.append("\n".join(block))

# Run
for t in targets:
    title = t.replace("is_", "").upper()
    run_logit(t, f"Event impact on {title}")

# =========================================================
# 6. Save report
# =========================================================
if results:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"\nSaved: {out_path}")
else:
    print("\nNo results to save (all skipped).")