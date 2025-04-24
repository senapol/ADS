import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler

from militaryaidstracker.correlation_aid_frontline import merged_df


aid_df = pd.read_csv("../data/cleaned/aid_categories_weekly.csv")
frontline_df = pd.read_csv("../data/cleaned/frontline_movement_with_distance.csv")

# merge weekly
aid_df["announcement_date"] = pd.to_datetime(aid_df["announcement_date"])
frontline_df["date"] = pd.to_datetime(frontline_df["date"])
frontline_df = frontline_df.rename(columns={"date": "announcement_date"})

merged_df = pd.merge(aid_df, frontline_weekly, on="announcement_date", how="inner")
merged_df = merged_df.sort_values("announcement_date")


selected_aid = "Heavy weapon"


scaler = MinMaxScaler()
normalized = scaler.fit_transform(merged_df[[selected_aid, "movement_km"]].fillna(0))
aid_series = normalized[:, 0]
move_series = normalized[:, 1]
dates = merged_df["announcement_date"].dt.strftime("%Y-%m-%d")


alignment = dtw.warping_path(aid_series, move_series)

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

