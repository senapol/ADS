import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError
# from scipy.signal import savgol_filter
# from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev, interp1d
from shapely.geometry import LineString, Point, MultiPoint, Polygon, mapping
from shapely.ops import nearest_points
# from scipy.spatial.distance import cdist
import json
import pyproj
# from functools import partial
from shapely.ops import transform
import warnings
warnings.filterwarnings('ignore')

ukraine_area = 603550 # area in kmsq

def detect_stable_frontline(df, date, window=60, eps_values=[0.08, 0.12, 0.16], min_samples_values=[3, 5, 7], weight_by_density=True):
    'Create a consensus frontline using multiple DBSCAN runs with different parameters'
    print(f'Creating consensus front line for {date}')
    
    start_date = date - pd.Timedelta(days=window)
    events = df[(df['date'] <= date) & (df['date'] >= start_date)].copy()
    
    if len(events) < 30:
        print(f"Not enough events in time window for {date}")
        return None, None
    
    combat_types = ['Battles', 'Armed Clash', 'Shelling/artillery/missile attack', 'Air/Drone Strike',
                    'Battle', 'Explosions/Remote violence', 'Russian Military Presence',
                    'Artillery/Missile Strike']
    
    is_combat = events['event_type'].apply(
        lambda x: any(combat_type.lower() in str(x).lower() for combat_type in combat_types)
    )
    combat_events = events[is_combat].copy()
    
    # if len(combat_events) < 30:
    #     print(f"Using all events as not enough military events")
    #     combat_events = events.copy()
    
    # Pre-filter to focus on combat zone using larger eps
    if len(combat_events) > 100:
        coords = combat_events[['longitude', 'latitude']].values
        rough_clustering = DBSCAN(eps=0.3, min_samples=10).fit(coords)
        combat_events['rough_cluster'] = rough_clustering.labels_
        
        # Find major clusters that likely form frontline
        cluster_counts = combat_events['rough_cluster'].value_counts()
        large_clusters = cluster_counts[cluster_counts > len(combat_events) * 0.05].index.tolist()
        
        if -1 in large_clusters:
            large_clusters.remove(-1) # remove noise
        
        if large_clusters:
            # Keep main combat clusters and a sample of noise points
            main_events = combat_events[combat_events['rough_cluster'].isin(large_clusters)]
            
            if len(combat_events[combat_events['rough_cluster'] == -1]) > 0:
                sample_size = min(100, len(combat_events[combat_events['rough_cluster'] == -1]))
                noise_sample = combat_events[combat_events['rough_cluster'] == -1].sample(
                    sample_size, random_state=42
                )
                combat_events = pd.concat([main_events, noise_sample])
            else:
                combat_events = main_events
                
            print(f"Filtered to {len(combat_events)} events in main combat zones")
    
    # Run multiple clusterings with different parameters to create a consensus
    all_frontline_points = []
    all_weights = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # Apply clustering with current parameters
                coords = combat_events[['longitude', 'latitude']].values
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
                tmp_events = combat_events.copy()
                tmp_events['cluster'] = clustering.labels_
                
                # Identify valid clusters (excluding noise labeled as -1)
                cluster_ids = sorted([cid for cid in set(clustering.labels_) if cid != -1])
                
                if len(cluster_ids) < 2:
                    print(f"Skipping eps={eps}, min_samples={min_samples}: not enough clusters")
                    continue
                
                # Calculate key points for each cluster
                cluster_points = []
                for cluster_id in cluster_ids:
                    try:
                        cluster_data = tmp_events[tmp_events['cluster'] == cluster_id]
                        
                        # Skip very small clusters
                        if len(cluster_data) < min_samples:
                            continue
                            
                        # Add centroid
                        centroid = (
                            cluster_data['longitude'].mean(),
                            cluster_data['latitude'].mean()
                        )
                        
                        # Use a weighted centroid for higher impact
                        for _ in range(3):  # Add centroid multiple times for higher weight
                            cluster_points.append({
                                'cluster_id': cluster_id,
                                'point_type': 'centroid',
                                'longitude': centroid[0],
                                'latitude': centroid[1],
                                'size': len(cluster_data),
                                'eps': eps,
                                'min_samples': min_samples
                            })
                        
                        # Add key perimeter points if enough data
                        if len(cluster_data) >= 5:
                            try:
                                # Make sure points are unique to avoid QHull errors
                                cluster_coords = cluster_data[['longitude', 'latitude']].values
                                unique_coords, unique_indices = np.unique(cluster_coords, axis=0, return_index=True)
                                
                                # Only proceed if we have enough unique points 
                                if len(unique_coords) >= 3:
                                    # Check if points are not collinear using SVD
                                    u, s, vh = np.linalg.svd(unique_coords - np.mean(unique_coords, axis=0))
                                    if s[1] > 1e-10:  # Second singular value should be non-zero for non-collinear points
                                        hull = ConvexHull(unique_coords)
                                        hull_vertices = unique_coords[hull.vertices]
                                        
                                        # Add extreme NESW points only
                                        north_idx = np.argmax(hull_vertices[:, 1])
                                        south_idx = np.argmin(hull_vertices[:, 1])
                                        east_idx = np.argmax(hull_vertices[:, 0]) 
                                        west_idx = np.argmin(hull_vertices[:, 0])
                                        
                                        for idx, name in [(north_idx, 'north'), (south_idx, 'south'), 
                                                         (east_idx, 'east'), (west_idx, 'west')]:
                                            point = hull_vertices[idx]
                                            # Only add if significantly different from centroid
                                            dist = np.sqrt(np.sum((point - centroid)**2))
                                            if dist > 0.01:  # ~1km
                                                cluster_points.append({
                                                    'cluster_id': cluster_id,
                                                    'point_type': name,
                                                    'longitude': point[0],
                                                    'latitude': point[1],
                                                    'size': len(cluster_data),
                                                    'eps': eps,
                                                    'min_samples': min_samples
                                                })
                                    else:
                                        print(f"Points in cluster {cluster_id} are collinear, skipping hull")
                                else:
                                    print(f"Not enough unique points in cluster {cluster_id}, skipping hull")
                            except QhullError as qe:
                                print(f"QHull error with cluster {cluster_id}: {qe}")
                                # Fall back to simple min/max points
                                longitude_min = cluster_data['longitude'].min()
                                longitude_max = cluster_data['longitude'].max()
                                latitude_min = cluster_data['latitude'].min()
                                latitude_max = cluster_data['latitude'].max()
                                
                                # Add these extrema points
                                extrema_points = [
                                    (longitude_min, cluster_data.loc[cluster_data['longitude'].idxmin(), 'latitude']),
                                    (longitude_max, cluster_data.loc[cluster_data['longitude'].idxmax(), 'latitude']),
                                    (cluster_data.loc[cluster_data['latitude'].idxmin(), 'longitude'], latitude_min),
                                    (cluster_data.loc[cluster_data['latitude'].idxmax(), 'longitude'], latitude_max)
                                ]
                                
                                for i, (lon, lat) in enumerate(extrema_points):
                                    # Check if point is different from centroid
                                    if abs(lon - centroid[0]) > 0.005 or abs(lat - centroid[1]) > 0.005:
                                        cluster_points.append({
                                            'cluster_id': cluster_id,
                                            'point_type': f'extrema_{i}',
                                            'longitude': lon,
                                            'latitude': lat,
                                            'size': len(cluster_data),
                                            'eps': eps,
                                            'min_samples': min_samples
                                        })
                            except Exception as e:
                                print(f"Other error with hull for cluster {cluster_id}: {e}")
                    except Exception as e:
                        print(f"Error processing cluster {cluster_id}: {e}")
                
                # Create a DataFrame from the points for this parameter set
                if len(cluster_points) < 3:
                    print(f"Skipping eps={eps}, min_samples={min_samples}: not enough key points")
                    continue
                
                points_df = pd.DataFrame(cluster_points)
                
                # Order points north-to-south for Ukraine's frontline shape
                # Ukrainian frontline generally runs north-south with a curve
                sorted_points = points_df.sort_values('latitude', ascending=False)
                
                # Extract coordinates with weights based on cluster size
                coords = sorted_points[['longitude', 'latitude']].values
                
                # Calculate weights
                if weight_by_density:
                    weights = np.array(sorted_points['size'])
                    weights = weights / weights.sum()  # Normalize
                    weights = weights * len(weights)   # Scale back up
                else:
                    weights = np.ones(len(coords))
                
                # Apply smoothing to the points for this parameter set
                try:
                    # Create a weighted parametric spline
                    t = np.linspace(0, 1, len(coords))
                    tck, u = splprep([coords[:, 0], coords[:, 1]], u=t, s=0.3)
                    new_t = np.linspace(0, 1, 50)  # Standardize to 50 points
                    smoothed_coords = np.column_stack(splev(new_t, tck))
                    
                    # Save these points for later averaging
                    all_frontline_points.append(smoothed_coords)
                    weight_value = 1.0 / (eps * min_samples)  # Higher weight for smaller eps & min_samples
                    all_weights.append(weight_value)
                except Exception as e:
                    print(f"Error in spline smoothing for eps={eps}, min_samples={min_samples}: {e}")
                
            except Exception as e:
                print(f"Error with parameters eps={eps}, min_samples={min_samples}: {e}")
    
    if not all_frontline_points:
        print("Failed to generate any valid frontlines")
        return None, combat_events
    
    # Normalize weights
    all_weights = np.array(all_weights)
    all_weights = all_weights / all_weights.sum()
    
    # Create an averaged consensus frontline
    n_points = 50  # All smoothed_coords should already have this length
    consensus_frontline = np.zeros((n_points, 2))
    
    for i, (frontline, weight) in enumerate(zip(all_frontline_points, all_weights)):
        # Skip if the frontline has different number of points
        if len(frontline) != n_points:
            continue
        
        # Weighted addition
        consensus_frontline += frontline * weight.reshape(-1, 1)
    
    # Make sure we have a valid frontline
    if np.all(consensus_frontline == 0):
        print("Failed to create a consensus frontline")
        return None, combat_events
    
    # Further smooth the consensus frontline for a cohesive shape
    try:
        # Create a parametric spline for final smoothing
        t = np.linspace(0, 1, n_points)
        tck, u = splprep([consensus_frontline[:, 0], consensus_frontline[:, 1]], u=t, s=0.3)
        
        # Sample more points for the final curve
        n_final = 100
        new_t = np.linspace(0, 1, n_final)
        final_frontline = np.column_stack(splev(new_t, tck))
        
        # Check for self-intersections
        line = LineString(final_frontline)
        if not line.is_simple:
            print("Detected self-intersections, applying simplification")
            line = line.simplify(0.01)
            final_frontline = np.array(line.coords)
    
    except Exception as e:
        print(f"Error in final smoothing: {e}")
        final_frontline = consensus_frontline
    
    return final_frontline, combat_events

def create_territory_polygon(frontline_coords, border_coords, eastern_border_lon=40.5):
    """
    Create a polygon representing the territory between the frontline and eastern border.
    Parameters:
    - frontline_coords: array of [lon, lat] frontline coordinates
    - border_coords: tuple of (lats, lons) arrays for Ukraine border
    - eastern_border_lon: longitude of default eastern border if border_coords is None
    Returns:
    - Polygon object representing the territory
    - Area in sqkm
    """
    if frontline_coords is None or len(frontline_coords) < 3:
        print("Not enough frontline points to create territory polygon")
        return None, 0
    
    # Create a LineString from the frontline
    frontline = LineString(frontline_coords)
    
    # Get north and south extent of the frontline
    north_lat = max(frontline_coords[:, 1])
    south_lat = min(frontline_coords[:, 1])
    
    # Find eastern border segments
    if border_coords is not None:
        try:
            lats, lons = border_coords
            border_points = np.column_stack([lons, lats])
            border = LineString(border_points)
            
            # Find eastern segments of the border
            # Simplify by finding points with longitude > certain threshold
            east_lon_threshold = np.median(frontline_coords[:, 0])  # Use median frontline longitude
            eastern_border_points = []
            
            for i, lon in enumerate(lons):
                if lon > east_lon_threshold:
                    eastern_border_points.append((lon, lats[i]))
            
            # If we found enough eastern border points, create a LineString
            if len(eastern_border_points) >= 2:
                eastern_border = LineString(eastern_border_points)
            else:
                # Fallback: create a vertical line at eastern_border_lon
                print("Not enough eastern border points found, using fallback")
                eastern_border = LineString([
                    (eastern_border_lon, north_lat + 0.5),
                    (eastern_border_lon, south_lat - 0.5)
                ])
        except Exception as e:
            print(f"Error processing border: {e}, using fallback")
            # Fallback to a vertical line
            eastern_border = LineString([
                (eastern_border_lon, north_lat + 0.5),
                (eastern_border_lon, south_lat - 0.5)
            ])
    else:
        # If no border provided, use a vertical line at eastern_border_lon as eastern border
        eastern_border = LineString([
            (eastern_border_lon, north_lat + 0.5),
            (eastern_border_lon, south_lat - 0.5)
        ])
    
    # Extract coordinates from border and frontline
    eastern_coords = list(eastern_border.coords)
    frontline_coords_list = list(frontline.coords)
    
    # Find northmost and southmost points
    try:
        # Find the northmost point on the frontline (maximum latitude)
        north_frontline_idx = np.argmax([p[1] for p in frontline_coords_list])
        north_frontline_point = frontline_coords_list[north_frontline_idx]
        
        # Find the northmost point on the eastern border
        north_eastern_idx = np.argmax([p[1] for p in eastern_coords])
        north_eastern_point = eastern_coords[north_eastern_idx]
        
        # Find the southmost point on the frontline (minimum latitude)
        south_frontline_idx = np.argmin([p[1] for p in frontline_coords_list])
        south_frontline_point = frontline_coords_list[south_frontline_idx]
        
        # Find the southmost point on the eastern border
        south_eastern_idx = np.argmin([p[1] for p in eastern_coords])
        south_eastern_point = eastern_coords[south_eastern_idx]
    except Exception as e:
        print(f"Error finding extremal points: {e}")
        return None, 0
    
    # Create north and south connectors
    north_connector_coords = [north_frontline_point, north_eastern_point]
    south_connector_coords = [south_frontline_point, south_eastern_point]
    
    # Make sure the polygon is properly oriented for closure
    # Frontline should be south-to-north if eastern border is north-to-south
    frontline_is_northward = frontline_coords_list[0][1] < frontline_coords_list[-1][1]
    eastern_is_southward = eastern_coords[0][1] > eastern_coords[-1][1]
    
    # Orient frontline from south to north if needed
    if not frontline_is_northward:
        frontline_coords_list = frontline_coords_list[::-1]
    
    # Orient eastern border from north to south if needed
    if not eastern_is_southward:
        eastern_coords = eastern_coords[::-1]
    
    # Combine all coordinates into a closed polygon
    # Order: frontline (south to north), north connector, eastern border (north to south), south connector
    polygon_coords = frontline_coords_list + north_connector_coords + eastern_coords + south_connector_coords[::-1]
    
    # Create a Shapely Polygon
    try:
        territory_polygon = Polygon(polygon_coords)
        
        # Make sure the polygon is valid
        if not territory_polygon.is_valid:
            print("Invalid polygon, attempting to fix")
            territory_polygon = territory_polygon.buffer(0)  # This often fixes invalid polygons
            
            if not territory_polygon.is_valid:
                print("Could not create a valid polygon")
                return None, 0
        
        # Calculate area in square kilometers
        # First convert to a projected CRS suitable for Ukraine
        # UTM zone 36N is good for Ukraine
        proj = pyproj.Transformer.from_crs(
            'EPSG:4326',  # WGS84
            'EPSG:32636',  # UTM zone 36N
            always_xy=True
        )
        
        # Apply the projection to the polygon
        projected_polygon = transform(proj.transform, territory_polygon)
        
        # Calculate area in square meters, then convert to square kilometers
        area_sq_km = projected_polygon.area / 1e6
        
        return territory_polygon, area_sq_km
    except Exception as e:
        print(f"Error creating polygon: {e}")
        return None, 0

def visualise_frontline_with_territory(frontline_coords, combat_events, territory_polygon, 
                                      area_sq_km, date, border_coords=None):
    """
    Create a visualisation showing the frontline, events, and controlled territory.
    Parameters:
    - frontline_coords: Array of [lon, lat] frontline coordinates
    - combat_events: DataFrame of combat events
    - territory_polygon: Shapely Polygon object representing controlled territory
    - area_sq_km: Area of the territory in square kilometers
    - date: Date of the analysis
    - border_coords: Optional tuple of (lats, lons) for the country border
    """
    # Create a base figure with events
    fig = px.scatter_mapbox(
        combat_events,
        lat='latitude',
        lon='longitude',
        color='rough_cluster' if 'rough_cluster' in combat_events.columns else None,
        hover_name='location',
        hover_data={
            'date': True,
            'event_type': True
        },
        color_continuous_scale=px.colors.qualitative.Plotly,
        mapbox_style="carto-positron",
        zoom=6,
        center={"lat": 49.0, "lon": 32.0},
        height=800,
        width=1200,
        title=f'Frontline and Occupied Territory - {date.strftime("%Y-%m-%d")} - {area_sq_km:.2f} km²'
    )
    
    # Add country border if provided
    if border_coords is not None:
        lats, lons = border_coords
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='lines',
            line=dict(width=2, color='green'),
            name='Ukraine Border'
        ))
    
    # Add territory polygon
    if territory_polygon is not None:
        try:
            # Extract polygon coordinates
            polygon_coords = list(territory_polygon.exterior.coords)
            polygon_lons = [p[0] for p in polygon_coords]
            polygon_lats = [p[1] for p in polygon_coords]
            
            # Add as a filled area
            fig.add_trace(go.Scattermapbox(
                lat=polygon_lats,
                lon=polygon_lons,
                mode='lines',
                line=dict(width=1, color='rgba(255,0,0,0.8)'),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                name=f'Occupied Territory ({area_sq_km:.2f} km²)'
            ))
        except Exception as e:
            print(f"Error adding territory polygon to visualisation: {e}")
    
    # Add the frontline
    fig.add_trace(go.Scattermapbox(
        lat=frontline_coords[:, 1],
        lon=frontline_coords[:, 0],
        mode='lines',
        line=dict(width=4, color='red'),
        name='Frontline'
    ))
    
    # Save visualization
    filename = f"Frontline_with_Territory_{date.strftime('%Y_%m_%d')}.html"
    fig.write_html(filename)
    print(f"Saved visualisation to {filename}")
    
    return fig

def analyse_territorial_changes_over_time(df, start_date='2022-02-01', end_date='2023-12-31', 
                                         interval=1, border_coords=None, eastern_border_lon=40.5):
    "Returns dictionary of {date: {'frontline': coords, 'area': area_sq_km, 'polygon': territory_polygon}}"

    print("Analysing territorial changes over time")
    
    # Convert to datetime and filter to range
    df['date'] = pd.to_datetime(df['date'])
    date_range_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    date_range_df['month_year'] = date_range_df['date'].dt.strftime('%Y-%m')
    
    # Get unique months in the range
    months = sorted(date_range_df['month_year'].unique())
    selected_months = months[::interval]  # Take every 'interval' months
    
    # Store results for each time period
    results = {}
    
    for month in selected_months:
        # Get last day of the month
        month_data = date_range_df[date_range_df['month_year'] == month]
        last_date = month_data['date'].max()
        
        print(f"Processing {month}, end date: {last_date}")
        
        # Generate frontline
        try:
            frontline_coords, combat_events = detect_stable_frontline(
                df, 
                last_date, 
                window=60,
                eps_values=[0.08, 0.12, 0.16],
                min_samples_values=[3, 5, 7]
            )
            
            if frontline_coords is None or combat_events is None:
                print(f"Skipping {month} - couldn't generate frontline")
                continue
            
            # Create territory polygon and calculate area
            territory_polygon, area_sq_km = create_territory_polygon(
                frontline_coords, 
                border_coords,
                eastern_border_lon
            )
            
            if territory_polygon is None:
                print(f"Skipping {month} - couldn't create territory polygon")
                continue
            
            # Visualize frontline and territory
            fig = visualise_frontline_with_territory(
                frontline_coords, 
                combat_events, 
                territory_polygon, 
                area_sq_km, 
                last_date, 
                border_coords
            )
            
            # Store results
            results[month] = {
                'date': last_date,
                'frontline': frontline_coords,
                'area': area_sq_km,
                'polygon': territory_polygon
            }
            
        except Exception as e:
            print(f"Error processing {month}: {e}")
            continue
    
    # Create a summary visualszation of territorial changes
    if results:
        # Create a line chart of area changes
        months = list(results.keys())
        areas = [results[month]['area'] for month in months]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=areas,
            mode='lines+markers',
            name='Occupied Territory (km²)'
        ))
        
        fig.update_layout(
            title='Occupied Territory in Ukraine Over Time',
            xaxis_title='Month',
            yaxis_title='Area (square kilometers)',
            height=600,
            width=1000
        )
        
        fig.write_html('Territorial_Changes_Summary.html')
        print("Saved territorial changes summary to Territorial_Changes_Summary.html")
    
    return results

if __name__ == "__main__":
    # Load data
    print('Loading combined data')
    df = pd.read_csv('data/combined_events.csv')
    # Convert dates (assumes European format day-month-year)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    
    # Load border if available
    try:
        with open('data/border.json', 'r') as file:
            data = json.load(file)
        
        coords = data['features'][0]['geometry']['coordinates'][0][0]
        lats = [coord[1] for coord in coords]
        lons = [coord[0] for coord in coords]
        border_coords = (lats, lons)
    except Exception as e:
        print(f"Border file not loaded: {e}")
        border_coords = None
        
    # Set eastern border longitude if border file is not available
    eastern_border_lon = 40.5  # Approximate eastern border of Ukraine
    
    # Option 1: Analyze a specific date
    key_date = pd.Timestamp('2023-08-15')
    
    # Generate frontline
    frontline_coords, combat_events = detect_stable_frontline(df, key_date)
    
    if frontline_coords is not None:
        # Create territory polygon and calculate area
        territory_polygon, area_sq_km = create_territory_polygon(
            frontline_coords, 
            border_coords,
            eastern_border_lon
        )
        
        # Visualize frontline and territory
        if territory_polygon is not None:
            visualise_frontline_with_territory(
                frontline_coords, 
                combat_events, 
                territory_polygon, 
                area_sq_km, 
                key_date, 
                border_coords
            )
            print(f"Occupied territory area: {area_sq_km:.2f} square kilometers")
    
    # Option 2: Analyze territorial changes over time
    # Uncomment to run this time-intensive analysis
    results = analyse_territorial_changes_over_time(
        df,
        start_date='2022-01-01', 
        end_date='2024-12-31', 
        interval=1,
        border_coords=border_coords,
        eastern_border_lon=eastern_border_lon
    )

months = []
dates = []
areas = []
percentages = []

# Ukraine's total area in square kilometers
ukraine_area = 603550

# Process each month's data
for month, data in results.items():
    try:
        # Extract date and area
        date = data['date']
        area = data['area']
        
        # Calculate percentage
        percent = (area / ukraine_area) * 100
        
        # Append to lists
        months.append(month)
        dates.append(date)
        areas.append(area)
        percentages.append(percent)
    except (KeyError, TypeError) as e:
        print(f"Error processing month {month}: {e}")

# Create DataFrame
df = pd.DataFrame({
    'month_year': months,
    'date': dates,
    'area_sq_km': areas,
    'percent_of_ukraine': percentages
})

# Sort by date
df = df.sort_values('date')

# Save to CSV
df.to_csv('data/frontline_area_output.csv', index=False)
print(f"Saved territorial results to csv")
