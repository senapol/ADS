import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Point, LineString
import plotly.graph_objects as go
import datetime
from scipy.ndimage import gaussian_filter1d
import requests
import zipfile
import os
import argparse


def download_ukraine_border():
    """
    Download high-resolution Ukraine border data.
    """
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_UKR_shp.zip"
    local_zip = "gadm41_UKR_shp.zip"

    if not os.path.exists("gadm_ukraine/gadm41_UKR_0.shp"):
        os.makedirs("gadm_ukraine", exist_ok=True)
        response = requests.get(url)
        with open(local_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall("gadm_ukraine")
        os.remove(local_zip)

    return gpd.read_file("gadm_ukraine/gadm41_UKR_0.shp")


def generate_precise_border_nodes(ukraine_border, num_nodes=20000):
    """
    Generate highly precise border nodes using advanced sampling techniques.
    """
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


def load_acled_data():
    """
    Load and preprocess ACLED dataset.
    """
    acled = pd.read_csv('ACLED_Ukraine_Reduced.csv')
    acled['event_date'] = pd.to_datetime(acled['event_date'])
    acled = acled[acled['event_date'] >= pd.Timestamp('2022-02-24')]
    acled['week'] = acled['event_date'].dt.to_period('W').apply(lambda r: r.start_time)

    current_date = acled['event_date'].max()
    acled['age_weeks'] = (current_date - acled['event_date']).dt.days / 7
    acled['weight'] = np.exp(-0.2 * acled['age_weeks'])

    return acled


def compute_weekly_frontline(acled, frontline_nodes, ukraine_border,
                             decay_lambda=0.2, search_radius=0.15, offset_factor=0.4):
    """
    Compute weekly frontline positions based on ACLED data.
    """
    weeks = sorted(acled['week'].unique())
    node_history = []
    current_nodes = frontline_nodes.copy()

    border_geom = ukraine_border.geometry.iloc[0]
    if border_geom.geom_type == 'MultiPolygon':
        border_geom = max(border_geom.geoms, key=lambda poly: poly.area)

    for week in weeks:
        print(f"Processing week: {week}")
        weekly_points = acled[acled['week'] <= week].copy()

        new_lat, new_lon = [], []

        for _, node in current_nodes.iterrows():
            distances = np.sqrt((weekly_points['latitude'] - node['latitude']) ** 2 +
                                (weekly_points['longitude'] - node['longitude']) ** 2)
            nearby = weekly_points[distances < search_radius]

            if not nearby.empty:
                offset_lat = ((nearby['latitude'] - node['latitude']) * nearby['weight']).sum() / nearby['weight'].sum()
                offset_lon = ((nearby['longitude'] - node['longitude']) * nearby['weight']).sum() / nearby['weight'].sum()

                new_point = Point(node['longitude'] + offset_factor * offset_lon, node['latitude'] + offset_factor * offset_lat)

                if border_geom.contains(new_point):
                    new_lat.append(new_point.y)
                    new_lon.append(new_point.x)
                else:
                    new_lat.append(node['latitude'])
                    new_lon.append(node['longitude'])
            else:
                new_lat.append(node['latitude'])
                new_lon.append(node['longitude'])

        new_lat = gaussian_filter1d(new_lat, sigma=3)
        new_lon = gaussian_filter1d(new_lon, sigma=3)

        current_nodes['latitude'], current_nodes['longitude'] = new_lat, new_lon
        current_nodes['week'] = week
        node_history.append(current_nodes.copy())

    return node_history


def plot_frontline_with_slider(node_history, acled, output_file):
    """
    Create an interactive Plotly map with slider to show frontline progression.
    """
    fig = go.Figure()

    for i, week_nodes in enumerate(node_history):
        week_nodes = week_nodes.copy()
        fig.add_trace(go.Scattermapbox(
            lat=week_nodes['latitude'],
            lon=week_nodes['longitude'],
            mode='lines',
            line=dict(width=2, color='red'),
            name=str(week_nodes['week'].iloc[0]),
            visible=(i == 0)
        ))

        week_points = acled[acled['week'] == week_nodes['week'].iloc[0]]
        if not week_points.empty:
            fig.add_trace(go.Scattermapbox(
                lat=week_points['latitude'],
                lon=week_points['longitude'],
                mode='markers',
                marker=dict(size=week_points['weight'] * 25, color='orange', opacity=0.4),
                showlegend=False,
                visible=(i == 0)
            ))

    steps = []
    for i in range(len(node_history)):
        step = dict(method="update", args=[{"visible": [False] * len(fig.data)}])
        step["args"][0]["visible"][i * 2] = True
        step["args"][0]["visible"][i * 2 + 1] = True if i * 2 + 1 < len(fig.data) else False
        step["label"] = str(node_history[i]['week'].iloc[0].date())
        steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Week: "}, steps=steps)],
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center={"lat": 49, "lon": 32},
        height=800
    )

    fig.write_html(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offset_factor', type=float, default=0.4, help='Offset factor for node adjustment')
    parser.add_argument('--search_radius', type=float, default=0.15, help='Search radius for event influence')
    parser.add_argument('--decay_lambda', type=float, default=0.2, help='Decay lambda for weight calculation')
    parser.add_argument('--output_file', type=str, default='frontline_map.html', help='Output HTML file')
    args = parser.parse_args()

    ukraine = download_ukraine_border()
    frontline_nodes = generate_precise_border_nodes(ukraine, num_nodes=20000)
    acled = load_acled_data()
    node_history = compute_weekly_frontline(acled, frontline_nodes, ukraine, decay_lambda=args.decay_lambda,
                                            search_radius=args.search_radius, offset_factor=args.offset_factor)

    plot_frontline_with_slider(node_history, acled, args.output_file)


if __name__ == "__main__":
    main()
