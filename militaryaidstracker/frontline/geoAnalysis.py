import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Load the dataset
file_path = "data/cleaned/cleaned_frontline_events.csv"
df = pd.read_csv(file_path)

# Convert 'date' to datetime format and sort by date
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

# Haversine function to compute distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

df['prev_latitude'] = df['latitude'].shift(1)
df['prev_longitude'] = df['longitude'].shift(1)
df['movement_km'] = df.apply(lambda row: haversine(row['prev_latitude'], row['prev_longitude'],
                                                   row['latitude'], row['longitude'])
                             if not pd.isnull(row['prev_latitude']) else np.nan, axis=1)


df.drop(columns=['prev_latitude', 'prev_longitude'], inplace=True)
output_file_path = "data/cleaned/frontline_movement_with_distance.csv"
df.to_csv(output_file_path, index=False)

print(f"File saved successfully: {output_file_path}")
