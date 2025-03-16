import numpy as np
from geopy.distance import geodesic
import pandas as pd

file_path_events = "data/filtered_frontline_events.csv"
df_events = pd.read_csv(file_path_events)

# Check if the dataset now contains data
df_events.info(), df_events.head()

# Convert 'date' to datetime format
df_events['date'] = pd.to_datetime(df_events['date'])

# Filter for relevant frontline events (Battles & Explosions)
frontline_events = df_events[df_events['event_type'].isin(["Battles", "Explosions/Remote violence"])]

# Sort by date
frontline_events = frontline_events.sort_values(by="date")

# Group by date and compute the daily frontline centroid (average lat/lon)
frontline_movement = frontline_events.groupby('date').agg(
    avg_latitude=('latitude', 'mean'),
    avg_longitude=('longitude', 'mean'),
    event_count=('event_type', 'count')  # Number of battle-related events per day
).reset_index()

# Compute movement distance (distance between successive frontlines)
def compute_distance(lat1, lon1, lat2, lon2):
    """Calculate geographical distance in km between two lat/lon coordinates."""
    if np.isnan(lat1) or np.isnan(lon1) or np.isnan(lat2) or np.isnan(lon2):
        return np.nan
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Apply distance computation to each row (comparing with previous day)
frontline_movement['distance_moved_km'] = frontline_movement.apply(
    lambda row: compute_distance(
        row['avg_latitude'], row['avg_longitude'],
        frontline_movement['avg_latitude'].shift(1), frontline_movement['avg_longitude'].shift(1)
    ), axis=1
)