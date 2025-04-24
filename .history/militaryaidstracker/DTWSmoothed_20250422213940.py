import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaller

# from militaryaidstracker.correlation_aid_frontline import merged_df

aid_df = pd.read_csv("data/cleaned/smoothed_aid_weekly.csv")
frontline_df = pd.read_csv("data/frontline_area_output.csv").head(36)
frontline_df["area_sq_km"] = -frontline_df["area_sq_km"]

categories = ['Humanitarian', 'Military equipment', 'Aviation and drones', 
              'Portable defence system', 'Heavy weapon', 'Financial', 'Uncategorised']

# merge weekly
aid_df["announcement_date"] = pd.to_datetime(aid_df["announcement_date"])
aid_df.loc[:, 'Total aid'] = aid_df[categories].sum(axis=1)
frontline_df["date"] = pd.to_datetime(frontline_df["date"])
frontline_df = frontline_df.rename(columns={"date": "announcement_date"})

merged_df = pd.merge(aid_df, frontline_df, on="announcement_date", how="inner")

selected_aid = "Portable defence system"

scaler = RobustScaller()
normalized = scaler.fit_transform(merged_df[[selected_aid, "area_sq_km"]].fillna(0))
aid_series = normalized[:, 0]
move_series = normalized[:, 1]
dates = merged_df["announcement_date"].dt.strftime("%Y-%m-%d")

alignment = dtw.warping_path(aid_series, move_series, window=6)

seen = set()
filtered_alignment = []
for i, j in alignment:
    if i not in seen and j not in seen:
        filtered_alignment.append((i,j))
        seen.add(i)
        seen.add(j)

plt.figure(figsize=(16, 5))
for i, j in alignment:
    plt.plot([i, j], [aid_series[i], move_series[j]], color = 'lightgray', linewidth = 0.6)

plt.plot(aid_series, label=f"{selected_aid} (normalised)", color = 'orange')
plt.plot(move_series, label ="Frontline Movement (normalised)", color='blue')

xticks_idx = np.linspace(0, len(dates)-1, 10, dtype=int)
plt.xticks(xticks_idx, dates.iloc[xticks_idx], rotation=45)

plt.title(f"DTW Alignment: {selected_aid} vs Frontline Movement (Weekly)")
plt.xlabel("Week")
plt.ylabel("Normalised Value")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

