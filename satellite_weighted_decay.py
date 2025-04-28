import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, LineString
import datetime
from scipy.ndimage import gaussian_filter1d
import requests
import zipfile
import os


def download_ukraine_border():
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_UKR_shp.zip"
    local_zip = "gadm41_UKR_shp.zip"

    if not os.path.exists("gadm_ukraine/gadm41_UKR_0.shp"):
        os.makedirs("gadm_ukraine", exist_ok=True)
        print("Downloading Ukraine border data...")
        response = requests.get(url)
        with open(local_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall("gadm_ukraine")
        os.remove(local_zip)
    
    print("Loading Ukraine border...")
    return gpd.read_file("gadm_ukraine/gadm41_UKR_0.shp")


def generate_precise_border_nodes(ukraine_border, num_nodes=20000):
    print("Generating border nodes...")
    border_geom = ukraine_border.geometry.iloc[0]

    if border_geom.geom_type == 'MultiPolygon':
        border_geom = max(border_geom.geoms, key=lambda poly: poly.area)

    if border_geom.geom_type == 'Polygon':
        exterior_coords = list(border_geom.exterior.coords)
    else:
        raise ValueError(f"Unexpected geometry type: {border_geom.geom_type}")

    border_line = LineString(exterior_coords)
    border_length = border_line.length
    border_nodes = [border_line.interpolate(float(i * border_length) / num_nodes) for i in range(num_nodes)]

    frontline_nodes = pd.DataFrame({
        'latitude': [point.y for point in border_nodes],
        'longitude': [point.x for point in border_nodes]
    })

    return frontline_nodes


def load_merged_events_data():
    print("Loading merged events data...")
    events = pd.read_csv('events_merged.csv')
    events['date'] = pd.to_datetime(events['date'])
    
    events = events[events['date'] >= pd.Timestamp('2022-02-24')]
    
    events['month'] = events['date'].dt.to_period('M').apply(lambda r: r.start_time)

    current_date = events['date'].max()
    events['age_months'] = (current_date - events['date']).dt.days / 30

    events['weight'] = np.exp(-0.6 * events['age_months'])

    return events


def compute_monthly_frontline(events, frontline_nodes, ukraine_border,
                            decay_lambda=0.2, search_radius=0.75):
    print("Computing monthly frontline movement...")
    months = sorted(events['month'].unique())
    node_history = []
    current_nodes = frontline_nodes.copy()

    border_geom = ukraine_border.geometry.iloc[0]
    if border_geom.geom_type == 'MultiPolygon':
        border_geom = max(border_geom.geoms, key=lambda poly: poly.area)

    for i, month in enumerate(months):
        print(f"Processing month {i+1}/{len(months)}: {month}")
        current_month = pd.to_datetime(month)
        start_period = current_month - pd.DateOffset(months=3)
        monthly_points = events[(events['date'] >= start_period) & (events['date'] <= current_month)].copy()

        new_lat, new_lon = [], []

        for _, node in current_nodes.iterrows():
            distances = np.sqrt((monthly_points['latitude'] - node['latitude']) ** 2 +
                                (monthly_points['longitude'] - node['longitude']) ** 2)
            nearby = monthly_points[distances < search_radius]

            if not nearby.empty:
                weight_sum = nearby['weight'].sum()
                if weight_sum > 0:
                    offset_lat = ((nearby['latitude'] - node['latitude']) * nearby['weight']).sum() / weight_sum
                    offset_lon = ((nearby['longitude'] - node['longitude']) * nearby['weight']).sum() / weight_sum

                    new_point = Point(node['longitude'] + 0.9 * offset_lon, node['latitude'] + 0.9 * offset_lat)

                    if border_geom.contains(new_point):
                        new_lat.append(new_point.y)
                        new_lon.append(new_point.x)
                    else:
                        new_lat.append(node['latitude'])
                        new_lon.append(node['longitude'])
                else:
                    new_lat.append(node['latitude'])
                    new_lon.append(node['longitude'])
            else:
                new_lat.append(node['latitude'])
                new_lon.append(node['longitude'])

        new_lat = gaussian_filter1d(new_lat, sigma=1)
        new_lon = gaussian_filter1d(new_lon, sigma=1)

        current_nodes['latitude'], current_nodes['longitude'] = new_lat, new_lon
        current_nodes['month'] = month
        node_history.append(current_nodes.copy())
        
        if i % 3 == 0:
            temp_df = pd.concat(node_history)
            temp_df.to_csv(f"merged_frontline_history_checkpoint_{i}.csv", index=False)

    return node_history


if __name__ == "__main__":
    ukraine = download_ukraine_border()
    frontline_nodes = generate_precise_border_nodes(ukraine, num_nodes=20000)
    events = load_merged_events_data()
    node_history = compute_monthly_frontline(events, frontline_nodes, ukraine)
    
    print("Saving node_history to 'merged_node_history.npy'...")
    np.save('merged_node_history.npy', node_history)
    print("Processing complete. Node history saved for merged dataset.")