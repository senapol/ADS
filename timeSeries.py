
import pandas as pd
from shapely.geometry import LineString, Polygon
from shapely.ops import transform
import pyproj
import numpy as np
import ast

def create_territory_polygon(frontline_coords, eastern_border_lon=40.5):
    if len(frontline_coords) < 3:
        return None, 0
    frontline_coords = sorted(frontline_coords, key=lambda x: x[1])
    south = [eastern_border_lon, frontline_coords[0][1]]
    north = [eastern_border_lon, frontline_coords[-1][1]]
    poly_coords = frontline_coords + [north, south, frontline_coords[0]]
    polygon = Polygon(poly_coords)
    proj = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    projected_polygon = transform(proj.transform, polygon)
    area_sqkm = projected_polygon.area / 1e6
    return polygon, area_sqkm

# Load datasets
major_nodes_df = pd.read_csv("L_final_east_border_nodes.csv")
frontline_nodes_df = pd.read_csv("frontline_nodes.csv")

# Process major nodes
major_areas = []
for _, row in major_nodes_df.iterrows():
    week = row['week']
    try:
        nodes = ast.literal_eval(row['nodes'])
        _, area = create_territory_polygon(nodes)
        major_areas.append({"week": week, "area_sq_km": area})
    except:
        major_areas.append({"week": week, "area_sq_km": None})
major_result = pd.DataFrame(major_areas)
major_result['gain_loss_sq_km'] = major_result['area_sq_km'].diff()
major_result.to_csv("major_nodes_territory_area.csv", index=False)

# Process frontline nodes
frontline_nodes_df['Date'] = pd.to_datetime(frontline_nodes_df['Date'])
frontline_nodes_df['week'] = frontline_nodes_df['Date'].dt.to_period('W').apply(lambda r: r.start_time)
weekly_coords = (
    frontline_nodes_df.groupby('week')[['Longitude', 'Latitude']]
    .apply(lambda g: g[['Longitude', 'Latitude']].values.tolist())
    .reset_index(name='coords')
)

frontline_areas = []
for _, row in weekly_coords.iterrows():
    week = row['week'].strftime('%Y-%m-%d')
    try:
        _, area = create_territory_polygon(row['coords'])
        frontline_areas.append({"week": week, "area_sq_km": area})
    except:
        frontline_areas.append({"week": week, "area_sq_km": None})
frontline_result = pd.DataFrame(frontline_areas)
frontline_result['gain_loss_sq_km'] = frontline_result['area_sq_km'].diff()
frontline_result.to_csv("frontline_nodes_territory_area.csv", index=False)
