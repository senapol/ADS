import numpy as np
from shapely.geometry import LineString, Point, MultiPoint, Polygon, mapping
import pyproj
from shapely.ops import transform
import json

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