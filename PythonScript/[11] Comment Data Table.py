import os
import pandas as pd
import numpy as np

# =========================================================
# 1) Load race–lap dataset (already aggregated + event markers)
# =========================================================
file_path = '/Users/chohyunseo/Desktop/SocialData_SportsAnalysis/SocialData_SportsCommunity/AnalysisResults/f1_race_lap_analysis_with_event.xlsx'
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

print(f"Loaded: {file_path}")
print(f"Rows (race-lap): {len(df)}")

# =========================================================
# 2) Columns to analyze
# =========================================================
event_vars = ['ev_unexp', 'ev_resp', 'ev_out']
metrics = ['comment_count', 'avg_text_len', 'avg_time_gap']

missing_events = [c for c in event_vars if c not in df.columns]
missing_metrics = [c for c in metrics if c not in df.columns]

if missing_events:
    raise ValueError(f"Missing event columns: {missing_events}")
if missing_metrics:
    raise ValueError(f"Missing metric columns: {missing_metrics}")

# Ensure events are 0/1 ints
for c in event_vars:
    df[c] = df[c].fillna(0)
    # sometimes events may be floats; coerce safely
    df[c] = (df[c].astype(float) > 0).astype(int)

# =========================================================
# 3) Build summary table: mean(metrics) by event occurrence (0/1)
# =========================================================
print("\n[이벤트별 영향력 요약표 생성]")

summary_rows = []

for ev in event_vars:
    # group means
    g = df.groupby(ev)[metrics].mean(numeric_only=True)

    # also include counts for transparency
    counts = df[ev].value_counts(dropna=False).to_dict()
    n0 = int(counts.get(0, 0))
    n1 = int(counts.get(1, 0))

    # ensure both 0 and 1 rows exist (fill if not)
    if 0 not in g.index:
        g.loc[0] = [np.nan] * len(metrics)
    if 1 not in g.index:
        g.loc[1] = [np.nan] * len(metrics)

    g = g.sort_index()

    # reshape to long rows
    for occurred in [0, 1]:
        row = {
            'Event_Type': ev,
            'Is_Occurred': occurred,
            'N': n1 if occurred == 1 else n0,
        }
        for m in metrics:
            row[m] = float(g.loc[occurred, m]) if pd.notna(g.loc[occurred, m]) else np.nan
        summary_rows.append(row)

    # add diff row (1 - 0)
    diff_row = {
        'Event_Type': ev,
        'Is_Occurred': 'Diff(1-0)',
        'N': n1,  # keep N1 for reference
    }
    for m in metrics:
        v1 = g.loc[1, m]
        v0 = g.loc[0, m]
        diff_row[m] = float(v1 - v0) if (pd.notna(v1) and pd.notna(v0)) else np.nan
    summary_rows.append(diff_row)

export_df = pd.DataFrame(summary_rows)

# column order
export_df = export_df[['Event_Type', 'Is_Occurred', 'N', 'comment_count', 'avg_text_len', 'avg_time_gap']]

# =========================================================
# 4) Save
# =========================================================
out_dir = os.path.dirname(file_path)
save_path = os.path.join(out_dir, 'f1_event_impact_summary.xlsx')
export_df.to_excel(save_path, index=False)

print("\n저장 완료:", save_path)
print("\n미리보기:")
print(export_df.head(12))