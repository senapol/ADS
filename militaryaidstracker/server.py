import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import TimestampedGeoJson
from datetime import datetime
from math import radians, cos, sin, sqrt, atan2


def haversine(coord1, coord2):
    R = 6371  # Earth radius (km)
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def generate_starting_border_nodes(border_coords, spacing_km=20):
    nodes = []
    for i in range(len(border_coords) - 1):
        start, end = border_coords[i], border_coords[i + 1]
        dist = haversine((start[1], start[0]), (end[1], end[0]))
        num_points = max(int(dist // spacing_km), 1)
        for j in range(num_points + 1):
            lat = start[1] + (end[1] - start[1]) * (j / num_points)
            lon = start[0] + (end[0] - start[0]) * (j / num_points)
            nodes.append((lat, lon))
    return nodes

def compute_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance = haversine(points[i], points[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

def cluster_points(points, eps=30):
    if len(points) < 2:
        return []
    distance_matrix = compute_distance_matrix(points)
    clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
    clustering.fit(distance_matrix)
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label != -1:
            clusters.setdefault(label, []).append(points[i])
    centroids = [[np.mean([p[0] for p in cluster]), np.mean([p[1] for p in cluster])] for cluster in clusters.values() if cluster]
    return centroids

def build_frontline(date_str, border_nodes, df, window_days=60):
    date_timestamp = pd.Timestamp(date_str)
    start_window = date_timestamp - pd.Timedelta(days=window_days)
    relevant_types = ['Battles', 'Explosions/Remote violence', 'Strategic developments']
    mask = (df['event_date'] >= start_window) & (df['event_date'] <= date_timestamp) & (df['event_type'].isin(relevant_types))
    relevant_data = df[mask]
    battle_points = relevant_data[['latitude', 'longitude']].values.tolist()
    points = battle_points + border_nodes
    centroids = cluster_points(points, eps=30)
    if not centroids:
        return border_nodes
    west_point = min(border_nodes, key=lambda p: p[1])
    east_point = max(border_nodes, key=lambda p: p[1])
    frontline_points = [west_point] + sorted(centroids, key=lambda p: p[1]) + [east_point]
    return frontline_points



df = pd.read_csv("ACLED_Ukraine_Reduced.csv")
df['event_date'] = pd.to_datetime(df['event_date'])
df = df[df['event_date'] > '2021-11-30']

try:
    border_gdf = gpd.read_file("ukraine_border.geojson")
    border_polygon = border_gdf.geometry.iloc[0]
    ukraine_border = list(border_polygon.exterior.coords)
except:
    ukraine_border = [(22.0, 48.0), (40.0, 48.0), (40.0, 52.0), (22.0, 52.0), (22.0, 48.0)]

border_nodes = generate_starting_border_nodes(ukraine_border, spacing_km=20)


monthly_dates = []
date_strings = sorted(df['event_date'].dt.strftime('%Y-%m-%d').unique())
for date_str in date_strings:
    date = datetime.strptime(date_str, '%Y-%m-%d')
    if not monthly_dates or date.month != datetime.strptime(monthly_dates[-1], '%Y-%m-%d').month:
        monthly_dates.append(date_str)

frontlines = {}
for date_str in monthly_dates:
    line = build_frontline(date_str, border_nodes, df, window_days=60)
    if len(line) > 2:
        frontlines[date_str] = line


features = []
start_year = datetime.strptime(monthly_dates[0], '%Y-%m-%d').year
end_year = datetime.strptime(monthly_dates[-1], '%Y-%m-%d').year

for date, line in frontlines.items():
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    year_fraction = date_obj.year + date_obj.month / 12
    color_value = int(255 * (year_fraction - start_year) / max(1, end_year - start_year))
    color = f'#{255 - color_value:02x}{0:02x}{color_value:02x}'

    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[p[1], p[0]] for p in line]},
        "properties": {
            "times": [f"{date}T00:00:00Z"],
            "style": {"color": color, "weight": 4, "opacity": 0.8}
        }
    }
    features.append(feature)

frontline_geojson = {"type": "FeatureCollection", "features": features}

m = folium.Map(location=[49, 32], zoom_start=6)
if 'border_gdf' in locals():
    folium.GeoJson(border_gdf, style_function=lambda x: {"color": "black", "weight": 1}).add_to(m)

TimestampedGeoJson(frontline_geojson, period='P1M', add_last_point=False, duration='P1M', transition_time=300, auto_play=True, loop=True).add_to(m)
m.save("frontline_tracker_windowed.html")

print("✅ Saved map as frontline_tracker_windowed.html")