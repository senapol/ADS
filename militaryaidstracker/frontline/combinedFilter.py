import pandas as pd

events_file_path = "data/combined_events.csv"
events_df = pd.read_csv(events_file_path)

frontline_events = events_df[
    events_df["event_type"].isin(["Battles", "Explosions/Remote violence"])
]

frontline_events.to_csv("filtered_frontline_events.csv", index=False)
print(frontline_events.head())
