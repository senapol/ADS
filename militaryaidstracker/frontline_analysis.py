import os
import webbrowser

import requests
import zipfile
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from shapely import LineString
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union, nearest_points, transform
from pyproj import Transformer
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

# Constants
MIN_DISPLACEMENT = 5000  # 5km minimum
MAX_DISPLACEMENT = 20000  # 20km maximum
DISPLACEMENT_DECAY = 0.7  # 30% weekly decay of displacement effects


eastern_front_polygon = Polygon([
    (30.5, 50.5),  # NW of Kharkiv
    (38.0, 50.5),  # NE near Russia border
    (39.0, 46.5),  # SE coast above Mariupol
    (35.5, 45.0),  # Crimea edge
    (31.5, 46.5),  # Near Mykolaiv
    (30.5, 48.0),  # Back up to Dnipro
    (30.5, 50.5)   # Close the loop
])


def download_ukraine_border():
    """Download Ukraine border shapefile if not already present"""
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_UKR_shp.zip"
    local_zip = "gadm_ukraine.zip"
    if not os.path.exists("gadm_ukraine/gadm41_UKR_0.shp"):
        os.makedirs("gadm_ukraine", exist_ok=True)
        resp = requests.get(url)
        with open(local_zip, "wb") as f:
            f.write(resp.content)
        with zipfile.ZipFile(local_zip, 'r') as z:
            z.extractall("gadm_ukraine")
        os.remove(local_zip)
    return gpd.read_file("gadm_ukraine/gadm41_UKR_0.shp")



def extract_major_frontline(history, region_poly=eastern_front_polygon):
    filtered_history = []
    for df in history:
        mask = [region_poly.contains(Point(lon, lat)) for lon, lat in zip(df['longitude'], df['latitude'])]
        filtered_df = df[mask].copy()
        filtered_history.append(filtered_df)
    return filtered_history

def generate_precise_border_nodes(ukraine_border, num_nodes=5000):
    """Generate evenly spaced points along the border"""
    geom = ukraine_border.geometry.iloc[0]
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda p: p.area)
    exterior = geom.exterior
    L = exterior.length
    pts = [exterior.interpolate(i / num_nodes * L) for i in range(num_nodes)]
    return pd.DataFrame({
        'longitude': [p.x for p in pts],
        'latitude': [p.y for p in pts],
        'original_index': range(num_nodes)
    })


def load_acled_data(csv_path='ACLED_Ukraine_Reduced.csv'):
    """Load and preprocess ACLED data"""
    acled = pd.read_csv(csv_path)
    acled['event_date'] = pd.to_datetime(acled['event_date'])
    acled = acled[acled['event_date'] >= pd.Timestamp('2022-02-24')]

    # Calculate weights
    current_date = acled['event_date'].max()
    acled['age_weeks'] = (current_date - acled['event_date']).dt.days / 7
    acled['weight'] = np.exp(-0.05 * acled['age_weeks'])  # Slower decay

    acled['week'] = acled['event_date'].dt.to_period('W').apply(lambda r: r.start_time)
    return acled


def calculate_territory_change(current_front, new_front, ukraine_poly):
    """Calculate net area change between frontlines (km²)"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Create Polygons from frontlines
    current_line = LineString(zip(current_front['longitude'], current_front['latitude']))
    new_line = LineString(zip(new_front['longitude'], new_front['latitude']))

    # Create buffer polygons (1km buffer to calculate areas)
    current_poly = current_line.buffer(0.01)
    new_poly = new_line.buffer(0.01)

    # Calculate areas
    ukraine_area = transform(transformer.transform, ukraine_poly).area / 1e6

    # Area Ukraine gained
    gain_area = transform(transformer.transform,
                          new_poly.difference(current_poly).intersection(ukraine_poly)).area / 1e6

    # Area Ukraine lost
    loss_area = transform(transformer.transform,
                          current_poly.difference(new_poly).intersection(ukraine_poly)).area / 1e6

    return {
        'net_area_km2': gain_area - loss_area,
        'pct_change': ((gain_area - loss_area) / ukraine_area) * 100
    }


def compute_displacement(current_front, week_events, ukraine_poly):
    """Calculate new frontline positions based on event weights"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    inv_transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Convert events to meters
    event_coords = np.array([
        transformer.transform(lon, lat)
        for lon, lat in zip(week_events['longitude'], week_events['latitude'])
    ])
    event_weights = week_events['weight'].values

    if len(event_coords) == 0:
        return current_front.copy()

    # Find nearest events to each border point
    border_coords = np.array([
        transformer.transform(lon, lat)
        for lon, lat in zip(current_front['longitude'], current_front['latitude'])
    ])

    tree = cKDTree(event_coords)
    dists, idxs = tree.query(border_coords, k=5)

    # Calculate displacements
    displacements = np.zeros_like(border_coords)
    for i in range(len(border_coords)):
        valid = dists[i] < 100000  # Only consider events within 100km
        if not any(valid):
            continue

        total_weight = np.sum(event_weights[idxs[i][valid]])
        if total_weight == 0:
            continue

        # Weighted average direction toward events
        direction = np.sum(
            (event_coords[idxs[i][valid]] - border_coords[i]) *
            event_weights[idxs[i][valid], np.newaxis],
            axis=0
        ) / total_weight

        # Normalize and scale
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        displacement_mag = np.clip(
            np.sum(event_weights[idxs[i][valid]]) * 5000,  # Scale factor
            MIN_DISPLACEMENT,
            MAX_DISPLACEMENT
        )
        displacements[i] = direction * displacement_mag

    # Apply displacements
    new_front = current_front.copy()
    new_coords = border_coords + displacements * DISPLACEMENT_DECAY

    # Convert back to lat/lon
    for i, (x, y) in enumerate(new_coords):
        lon, lat = inv_transformer.transform(x, y)
        new_front.at[new_front.index[i], 'longitude'] = lon
        new_front.at[new_front.index[i], 'latitude'] = lat

    return new_front


def smooth_frontline(frontline_df, sigma=1.5):
    """Apply Gaussian smoothing to frontline"""
    lons = gaussian_filter1d(frontline_df['longitude'], sigma=sigma)
    lats = gaussian_filter1d(frontline_df['latitude'], sigma=sigma)
    smoothed = frontline_df.copy()
    smoothed['longitude'] = lons
    smoothed['latitude'] = lats
    return smoothed


def compute_weekly_frontline(acled, border_nodes_df, ukraine_poly):
    """Main frontline computation with territorial tracking"""
    history = []
    current_front = border_nodes_df.copy()
    current_front['week'] = acled['week'].min()  # Pre-war border

    for week in sorted(acled['week'].unique()):
        print(f"Processing {week}")
        week_events = acled[acled['week'] == week]

        # Calculate new frontline
        new_front = compute_displacement(current_front, week_events, ukraine_poly)
        new_front = smooth_frontline(new_front)
        new_front['week'] = week

        # Calculate territory change
        change = calculate_territory_change(current_front, new_front, ukraine_poly)
        new_front['territory_change'] = change['net_area_km2']
        new_front['advancing'] = 'Ukraine' if change['net_area_km2'] > 0 else 'Russia'

        history.append(new_front.copy())
        current_front = new_front

    return history


def plot_frontline_with_controls(node_history, acled):
    """Interactive plot with working play/pause buttons"""
    fig = go.Figure()

    # Create frames for animation
    frames = []
    for i, df in enumerate(node_history):
        week = df['week'].iloc[0]

        # Frontline trace
        frontline_trace = go.Scattermapbox(
            lat=df['latitude'],
            lon=df['longitude'],
            mode='lines',
            line=dict(width=3, color='red'),
            name=f"Frontline {week.date()}"
        )

        # Events trace
        week_events = acled[acled['week'] == week]
        event_trace = go.Scattermapbox(
            lat=week_events['latitude'],
            lon=week_events['longitude'],
            mode='markers',
            marker=dict(
                size=8,
                color='orange',
                opacity=0.7
            ),
            name=f"Events {week.date()}",
            showlegend=False
        )

        # Territory change annotation
        change = df['territory_change'].iloc[0]
        advancing = df['advancing'].iloc[0]
        annotation = dict(
            text=f"{advancing} advancing<br>{abs(change):.1f} km²",
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=12)
        )

        frames.append(go.Frame(
            data=[frontline_trace, event_trace],
            layout=dict(annotations=[annotation]),
            name=str(week.date())
        ))

    # Initial frame
    fig.add_trace(frames[0].data[0])
    fig.add_trace(frames[0].data[1])

    # Animation controls
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=5,
        mapbox_center={"lat": 48.5, "lon": 37},
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }
                    ]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ],
            "x": 0.1,
            "y": 0
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[frame.name], {"frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate"}],
                    "label": frame.name,
                    "method": "animate"
                }
                for frame in frames
            ],
            "transition": {"duration": 0},
            "x": 0.1,
            "len": 0.9,
            "currentvalue": {
                "prefix": "Week: ",
                "visible": True
            }
        }]
    )

    fig.frames = frames

    # Final layout adjustments
    fig.update_layout(
        height=800,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        title="Ukraine Frontline Evolution with Territorial Control"
    )

    html_file = 'ukraine_frontline_territorial2.html'
    fig.write_html(html_file)
    webbrowser.open(html_file)


if __name__ == '__main__':
    # Load data
    ukraine = download_ukraine_border()
    border_nodes = generate_precise_border_nodes(ukraine)
    ukraine_poly = ukraine.geometry.iloc[0]
    acled = load_acled_data()

    # Compute frontline evolution
    frontline_history = compute_weekly_frontline(acled, border_nodes, ukraine_poly)

    # Extract only the eastern active frontline
    major_frontline_history = extract_major_frontline(frontline_history)

    # Optionally re-plot and/or export
    plot_frontline_with_controls(major_frontline_history, acled)

    # Visualize
    plot_frontline_with_controls(frontline_history, acled)


    # Create a list of rows where each row is (week, [[lat1, lon1], [lat2, lon2], ..., [latN, lonN]])
    # Save structured CSV of major frontline only
    rows = []
    for df in major_frontline_history:
        week = df['week'].iloc[0].date()
        latlons = df[['latitude', 'longitude']].values.tolist()
        rows.append({'week': week, 'nodes': latlons})

    pd.DataFrame(rows).to_csv('major_eastern_frontline_nodes.csv', index=False)
    print("Saved major eastern frontline to 'major_eastern_frontline_nodes.csv'")
