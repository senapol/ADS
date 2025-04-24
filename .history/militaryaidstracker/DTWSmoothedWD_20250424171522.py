import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from scipy.signal import correlate

# from militaryaidstracker.correlation_aid_frontline import merged_df

aid_df = pd.read_csv("data/cleaned/smoothed_aid_weekly.csv")
frontline_df = pd.read_csv("data/WD_frontline_area_output_weekly.csv")
frontline_df["date"] = pd.to_datetime(frontline_df["date"])
frontline_df = frontline_df.rename(columns={"date": "announcement_date"})
frontline_df = frontline_df.groupby(pd.Grouper(key='announcement_date', freq='M')).agg({
    'area_sq_km': 'mean'
}).reset_index()
frontline_df["area_sq_km"] = -frontline_df["area_sq_km"]

frontline_df['area_sq_km'] = (
    frontline_df['area_sq_km']
    .diff()               # current month minus previous month
)

print(frontline_df.count())

categories = ['Humanitarian', 'Military equipment', 'Aviation and drones', 
              'Portable defence system', 'Heavy weapon', 'Financial', 'Uncategorised']

print(aid_df.count())

# merge weekly
aid_df["announcement_date"] = pd.to_datetime(aid_df["announcement_date"])
aid_df.loc[:, 'Total aid'] = aid_df[categories].sum(axis=1)

frontline_df["area_sq_km"] = frontline_df["area_sq_km"] - frontline_df["area_sq_km"].min()

merged_df = pd.merge(aid_df, frontline_df, on="announcement_date", how="inner")

selected_aid = "Aviation and drones"

scaler = MinMaxScaler()
normalized = scaler.fit_transform(merged_df[[selected_aid, "area_sq_km"]].fillna(0))
aid_series = normalized[:, 0]
move_series = normalized[:, 1]
dates = merged_df["announcement_date"].dt.strftime("%Y-%m-%d")

alignment = dtw.warping_path(aid_series, move_series, window=7)
dist = dtw.distance(    aid_series, move_series, window=7)

print(dist/len(alignment))

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

# aid_series  = merged_df[selected_aid]   / merged_df[selected_aid].max()
# move_series = -(merged_df["area_sq_km"] / merged_df["area_sq_km"].min()) + 1

results = []
for cat in categories + ['Total aid']:
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(merged_df[[cat, "area_sq_km"]].fillna(0))
    aid_series = normalized[:, 0]
    move_series = normalized[:, 1]
    dates = merged_df["announcement_date"].dt.strftime("%Y-%m-%d")

    # 2. compute DTW distance (not the full path)
    dist = dtw.distance(aid_series, move_series, window=7)

    aid_series = aid_series - aid_series.mean()
    move_series = move_series - move_series.mean()
    N  = len(aid_series)

    # full cross‐correlation
    corr_full = correlate(aid_series, move_series, mode="full")

    # build the lags array for mode="full"
    lags_full = np.arange(-N+1, N)

    # pick only lags between -5 and +5
    mask = (lags_full >= -6) & (lags_full <= +6)
    corr_small = corr_full[mask]
    lags_small = lags_full[mask]

    # find the best among those
    best_idx    = np.argmax(corr_small)
    best_lag     = lags_small[best_idx]
    best_corrval = corr_small[best_idx] / (np.std(aid_series)*np.std(move_series)*N)

    print(cat)
    print(f"Peak corr within ±5 at lag={best_lag}, r≈{best_corrval:.2f}")

    # 3. record it
    results.append({
        "category": cat,
        "dtw_distance": dist
    })

# 4. build a DataFrame and save
df_dist = pd.DataFrame(results).sort_values("dtw_distance")
df_dist.to_csv("dtw_distances_per_category.csv", index=False)
print(df_dist)
