import numpy as np
from shapely.geometry import LineString, Point, MultiPoint, Polygon, mapping
import pyproj
from shapely.ops import transform
import json
import pandas as pd
import ast

ukraine_area = 603550

def get_border_coords():
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
    return border_coords

def process_nn_frontline(nn_points, date):
    """
    Process neural network frontline data to extract and flatten coordinates.
    Parameters:
    - nn_data: Dictionary containing neural network output with dates as keys.
    - date: The specific date to extract frontline coordinates for.
    Returns:
    - List of [lon, lat] coordinates for the specified date.
    """
    if date not in nn_points:
        print(f'No frontline coords available for {date}')
        return None

    data = nn_points[date]
    coords = []
    for segment in data:
        for point in segment:
            if len(point) < 2:
                print(f"Invalid point {point}, skipping")
                continue
            coords.append((point[0], point[1]))  # [lon, lat]

    if not coords:
        print(f"No valid coordinates found for date {date}")
        return None
    return np.array(coords)

    

def create_territory_polygon(frontline_coords, eastern_border_lon=40.5):
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
    border_coords = get_border_coords()

    if frontline_coords is None or len(frontline_coords) < 3:
        print("Not enough frontline points to create territory polygon")
        return None, 0
    
    # create a LineString from the frontline
    frontline = LineString(frontline_coords)
    
    # get N and S extent of the frontline
    north_lat = max(frontline_coords[:, 1])
    south_lat = min(frontline_coords[:, 1])
    
    # find eastern border segments
    if border_coords is not None:
        try:
            lats, lons = border_coords
            border_points = np.column_stack([lons, lats])
            border = LineString(border_points)
            
            # simplify by finding points with longitude > certain threshold
            east_lon_threshold = np.median(frontline_coords[:, 0])  # use median frontline longitude
            eastern_border_points = []
            
            for i, lon in enumerate(lons):
                if lon > east_lon_threshold:
                    eastern_border_points.append((lon, lats[i]))
            
            # if enough eastern border points found, create a LineString
            if len(eastern_border_points) >= 2:
                eastern_border = LineString(eastern_border_points)
            else:
                # fallback: create vertical line at eastern_border_lon
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
        # if no border provided, use vertical line at eastern_border_lon as eastern border
        eastern_border = LineString([
            (eastern_border_lon, north_lat + 0.5),
            (eastern_border_lon, south_lat - 0.5)
        ])
    
    # extract coords from border and frontline
    eastern_coords = list(eastern_border.coords)
    frontline_coords_list = list(frontline.coords)
    
    # find northmost and southmost points
    try:
        # find the northmost point on the frontline (max lat)
        north_frontline_idx = np.argmax([p[1] for p in frontline_coords_list])
        north_frontline_point = frontline_coords_list[north_frontline_idx]
        
        # northmost point on the eastern border
        north_eastern_idx = np.argmax([p[1] for p in eastern_coords])
        north_eastern_point = eastern_coords[north_eastern_idx]
        
        # southmost point on the frontline (min lat)
        south_frontline_idx = np.argmin([p[1] for p in frontline_coords_list])
        south_frontline_point = frontline_coords_list[south_frontline_idx]
        
        # southmost point on the eastern border
        south_eastern_idx = np.argmin([p[1] for p in eastern_coords])
        south_eastern_point = eastern_coords[south_eastern_idx]
    except Exception as e:
        print(f"Error finding extremal points: {e}")
        return None, 0
    
    # create N and S connectors
    north_connector_coords = [north_frontline_point, north_eastern_point]
    south_connector_coords = [south_frontline_point, south_eastern_point]
    
    # make sure the polygon is properly oriented for closure
    # frontline should be S-N if eastern border is N-S
    frontline_is_northward = frontline_coords_list[0][1] < frontline_coords_list[-1][1]
    eastern_is_southward = eastern_coords[0][1] > eastern_coords[-1][1]
    
    # orient frontline S-N if needed
    if not frontline_is_northward:
        frontline_coords_list = frontline_coords_list[::-1]
    
    # orient eastern border from N-S if needed
    if not eastern_is_southward:
        eastern_coords = eastern_coords[::-1]
    
    # combine all coordinates into a closed polygon
    # order: frontline (south to north), north connector, eastern border (north to south), south connector
    polygon_coords = frontline_coords_list + north_connector_coords + eastern_coords + south_connector_coords[::-1]
    
    # create a Shapely Polygon
    try:
        territory_polygon = Polygon(polygon_coords)
        
        if not territory_polygon.is_valid:
            print("Invalid polygon, attempting to fix")
            territory_polygon = territory_polygon.buffer(0)  # fixes invalid polygons
            
            if not territory_polygon.is_valid:
                print("Could not create a valid polygon")
                return None, 0
        
        # calc area in sqkm
        # first convert to a projected CRS suitable for Ukraine
        # UTM zone 36N is good for Ukraine
        proj = pyproj.Transformer.from_crs(
            'EPSG:4326',  # WGS84
            'EPSG:32636',  # UTM zone 36N
            always_xy=True
        )
        
        # apply the projection to the polygon
        projected_polygon = transform(proj.transform, territory_polygon)
        
        # calc area in sqm, then convert to sqkm
        area_sq_km = projected_polygon.area / 1e6
        
        return territory_polygon, area_sq_km
    except Exception as e:
        print(f"Error creating polygon: {e}")
        return None, 0
    
import matplotlib.pyplot as plt

def plot_territory_polygon(polygon, title="Territory Polygon"):
    """
    Plot the territory polygon or multipolygon using matplotlib.
    Parameters:
    - polygon: Shapely Polygon or MultiPolygon object to plot.
    - title: Title of the plot.
    """
    if polygon is None:
        print("No polygon to plot")
        return

    plt.figure(figsize=(10, 8))

    if polygon.geom_type == 'Polygon':
        # Single polygon
        x, y = polygon.exterior.xy
        plt.plot(x, y, color='blue', linewidth=2, label='Territory Boundary')
        plt.fill(x, y, color='lightblue', alpha=0.5, label='Territory Area')
    elif polygon.geom_type == 'MultiPolygon':
        # MultiPolygon: iterate through individual polygons
        for poly in polygon.geoms:
            x, y = poly.exterior.xy
            plt.plot(x, y, color='blue', linewidth=2)
            plt.fill(x, y, color='lightblue', alpha=0.5)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def create_NN_frontline_area_csv():
    with open('data/nn_frontline_data.json', 'r') as file:
        nn_coords = json.load(file)
    
    results = {}
    for date in nn_coords.keys():
        print(f"Processing week starting: {date}")

        frontline_coords = process_nn_frontline(nn_coords, date)
        if frontline_coords is not None:
            territory_polygon, area_sq_km = create_territory_polygon(frontline_coords)
            if territory_polygon is not None:
                print(f"Territory polygon created with area: {area_sq_km} sqkm")
                percentage = (area_sq_km / ukraine_area) * 100
                plot_territory_polygon(territory_polygon, title=f"NN Territory Polygon for {date}")
                results[date] = {
                    'area_sq_km': area_sq_km,
                    'percent_of_ukraine': percentage,
                }
            else:
                print("Failed to create territory polygon")
        else:
            print("No frontline coordinates available")

    weeks = []
    areas = []
    percentages = []

    for date, result in results.items():
        print(f"Date: {date}, Area: {result['area_sq_km']} sqkm, Percentage of Ukraine: {result['percent_of_ukraine']:.2f}%")
        weeks.append(date)
        areas.append(result['area_sq_km'])
        percentages.append(result['percent_of_ukraine'])

    df = pd.DataFrame({
        'week_start': weeks,
        'area_sq_km': areas,
        'percent_of_ukraine': percentages
    })
    df = df.sort_values(by='week_start')
    df.to_csv('data/NN_frontline_area_output_weekly.csv', index=False)
    print(f"Saved weekly territorial results to CSV")

def create_WD_frontline_area_csv():
    df = pd.read_csv('data/major_eastern_frontline_nodes.csv')
    
    results = {}
    
    for _, row in df.iterrows():
        week = row['week']
        try:
            # Parse the nodes column
            nodes = ast.literal_eval(row['nodes'])

            if not isinstance(nodes, list) or len(nodes) < 3:
                print(f"Invalid nodes for week {week}, skipping")
                continue

            # Swap lat, lon to lon, lat
            coords = np.array([[point[1], point[0]] for point in nodes])

            # Create territory polygon and calculate area
            territory_polygon, area_sq_km = create_territory_polygon(coords)
            if territory_polygon is not None:
                percentage = (area_sq_km / ukraine_area) * 100
                plot_territory_polygon(territory_polygon, title=f"WD Territory Polygon for {week}")
                results[week] = {
                    'area_sq_km': area_sq_km,
                    'percent_of_ukraine': percentage,
                }
                print(f'Week {week}: Area: {area_sq_km} sqkm, Percentage of Ukraine: {percentage:.2f}%')
            else:
                print(f"Failed to create territory polygon for week {week}")
        except Exception as e:
            print(f"Error processing nodes for week {week}: {e}")
            continue

    # Prepare results for CSV
    weeks = []
    areas = []
    percentages = []

    for week, result in results.items():
        print(f"Week: {week}, Area: {result['area_sq_km']} sqkm, Percentage of Ukraine: {result['percent_of_ukraine']:.2f}%")
        weeks.append(week)
        areas.append(result['area_sq_km'])
        percentages.append(result['percent_of_ukraine'])

    # Save results to CSV
    df_output = pd.DataFrame({
        'week_start': weeks,
        'area_sq_km': areas,
        'percent_of_ukraine': percentages
    })
    df_output = df_output.sort_values(by='week_start')
    df_output.to_csv('data/WD_frontline_area_output_weekly.csv', index=False)
    print(f"Saved weekly territorial results to CSV")

create_WD_frontline_area_csv()