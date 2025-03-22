import pandas as pd

aid_df = pd.read_csv("data/aid_categories_weekly.csv")
frontline_df = pd.read_csv("data/frontline_movement_with_distance.csv")

# 1. date columns into datetime
aid_df["announcement_date"] = pd.to_datetime(aid_df["announcement_date"])
frontline_df["date"] = pd.to_datetime(frontline_df["date"])

# 2. group frontline movement data by week
frontline_weekly = (
    frontline_df.dropna(subset=["movement_km"])
    .groupby(pd.Grouper(key="date", freq="W"))
    .agg({"movement_km": "sum"})
    .reset_index()
    .rename(columns={"date": "announcement_date"})
)

# 3. merge aid & frontline movement data
merged_df = pd.merge(aid_df, frontline_weekly, on="announcement_date", how="inner")

# 4. aid categories
aid_categories = [
    "Humanitarian", "Military equipment", "Aviation and drones",
    "Portable defence system", "Heavy weapon", "Financial", "Uncategorised"
]

# 5. Pearson correlation
correlations = {
    category: merged_df[category].corr(merged_df["movement_km"])
    for category in aid_categories
}

# 6. convert
correlations_df = pd.DataFrame(list(correlations.items()), columns=["Aid Category", "Correlation with Frontline Movement"])
trend_df = merged_df[["announcement_date", "movement_km"] + aid_categories]

correlations_df = correlations_df.sort_values(by="Correlation with Frontline Movement", ascending=False).reset_index(drop=True)
trend_df.to_csv("weekly_aid_and_frontline_movement.csv", index=False)
correlations_df.to_csv("correlation_aid_vs_frontline.csv", index=False)
