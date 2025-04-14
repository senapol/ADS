# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 19:25:34 2025

@author: talia
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ipywidgets import interactive, IntSlider, fixed
from IPython.display import display
import datetime
import json
from shapely.geometry import Polygon, Point

def plot_classification_scatter_window(start_day_index, df):
    """
    Plots a scatter plot of events within a 5-day window based on event_date.
    - Red points if classification = 0
    - Blue points if classification = 1

    Parameters:
    start_day_index (int): The index of the starting day in the sorted unique event dates.
    df (pd.DataFrame): The DataFrame containing 'longitude', 'latitude', 'classification', and 'event_date'.

    Returns:
    None (Displays an interactive scatter plot)
    """

    # Sort and get unique event dates
    unique_dates = sorted(df['event_date'].unique())

    # Ensure the start index is valid
    if start_day_index >= len(unique_dates):
        print("Start index out of range")
        return

    # Define the 5-day window
    start_date = unique_dates[start_day_index]
    end_date = start_date + datetime.timedelta(days=4)

    # Filter data for the 5-day window
    filtered_df = df[(df['event_date'] >= start_date) & (df['event_date'] <= end_date)]
    #print(filtered_df)
    # Define colors
    colors = filtered_df['classification'].map({1: 'blue', 0: 'red'})

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['longitude'], filtered_df['latitude'], c=colors, alpha=0.6, edgecolors='k')

    # Formatting
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Events from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    plt.grid(True)

    # Show plot
    plt.show()


def plot_classification_scatter(df):
    """
    Plots a scatter graph using 'longitude' and 'latitude' from the DataFrame.
    - Red points if classification = 1
    - Blue points if classification = 0

    Parameters:
    df (pd.DataFrame): The DataFrame containing 'longitude', 'latitude', and 'classification' columns.

    Returns:
    None (Displays a scatter plot)
    """

    # Define colors based on classification values
    colors = df['classification'].map({1: 'red', 0: 'blue'})

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['longitude'], df['latitude'], c=colors, alpha=0.6, edgecolors='k')

    # Labels and title
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Event Locations by Classification")
    plt.grid(True)

    # Show plot
    plt.show()

def classify_events(df, dictionary):
    """
    Adds a 'classification' column to the DataFrame based on 'actor1' values.

    - If 'actor1' contains 'Russia' and the corresponding event type in the dictionary is True, classification = 1.
    - If 'actor1' contains 'Ukraine' and the corresponding event type in the dictionary is True, classification = -1.
    - Otherwise, classification = 0.
    - Rows where 'actor1' contains neither 'Russia' nor 'Ukraine' are removed.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    dictionary (dict): The dictionary that determines classification.

    Returns:
    pd.DataFrame: A modified DataFrame with the 'classification' column.
    """

    # Filter rows where 'actor1' contains 'Russia' or 'Ukraine'
    df = df[df['actor1'].str.contains('Russia|Ukraine', case=False, na=False)]

    # Apply classification logic
    def classify(row):
        event_type = row['sub_event_type']
        if 'Russia' in row['actor1']:
            if event_type in dictionary and dictionary[event_type]:
                return 1
            else:
                return 0
        elif 'Ukraine' in row['actor1']:
            if event_type in dictionary and dictionary[event_type]:
                return 0
            else:
                return 1

    df['classification'] = df.apply(classify, axis=1)

    return df

def duplicate_armed_clash_events(df):
    """
    Duplicates rows where 'sub_event_type' is 'Armed clash', swapping 'actor1' and 'actor2' in the duplicate.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with the duplicated 'Armed clash' events included.
    """

    # Filter for Armed clash events
    armed_clash_events = df[df['sub_event_type'] == 'Armed clash'].copy()

    # Swap actor1 and actor2 in the duplicated rows
    armed_clash_events[['actor1', 'actor2']] = armed_clash_events[['actor2', 'actor1']]

    # Append duplicated rows back to the original DataFrame
    df = pd.concat([df, armed_clash_events], ignore_index=True)

    return df

# Example usage:
# new_df = classify_events(new_df, behind_actor_1_lines)
# print(new_df)

df = pd.read_csv("data/ACLED_Ukraine_Reduced.csv")
df["event_date"] = pd.to_datetime(df["event_date"], dayfirst=True)


behind_actor_1_lines = {
    'Armed clash': True,
    'Abduction/forced disappearance': True,
    'Shelling/artillery/missile attack': False,
    'Headquarters or base established': True,
    'Disrupted weapons use': True,
    'Suicide bomb': False,
    'Non-state actor overtakes territory': True,
    'Attack': True,
    'Chemical weapon': False,
    'Government regains territory': True,
    'Air/drone strike': False
}

df = duplicate_armed_clash_events(df)

new_df = df[df['sub_event_type'].isin(behind_actor_1_lines.keys())]
#print(new_df)
new_df = classify_events(new_df, behind_actor_1_lines)
plot_classification_scatter(new_df)

dates = list(set(new_df['event_date']))
dates.sort()

# Load GeoJSON file

from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

with open("data/border.json", "r") as f:
    border_data = json.load(f)

# Extract all polygons from the GeoJSON data
polygons = []
for feature in border_data["features"]:
    geom = feature["geometry"]
    if geom["type"] == "MultiPolygon":
        for poly_coords in geom["coordinates"]:
            polygons.append(Polygon(poly_coords[0]))  # Assuming one outer ring in each MultiPolygon
    elif geom["type"] == "Polygon":
        polygons.append(Polygon(geom["coordinates"][0]))  # Assuming one outer ring in each Polygon

# Create a MultiPolygon object from the individual polygons
ukraine_polygon = MultiPolygon(polygons) # Changed this line to create MultiPolygon

# If ukraine_polygon is a MultiPolygon, get the exterior of the largest polygon
if ukraine_polygon.geom_type == 'MultiPolygon':
    # Get the polygon with the largest area
    # Iterate through the individual polygons within the MultiPolygon
    largest_polygon = max(ukraine_polygon.geoms, key=lambda p: p.area) #changed this line to iterate through polygons
    border_coords = list(largest_polygon.exterior.coords)
    ukraine_polygon = largest_polygon
else:
    # If it's a Polygon, get the exterior directly
    border_coords = list(ukraine_polygon.exterior.coords)

# Convert to a list of (longitude, latitude) tuples
border_coords = [(x, y) for x, y in border_coords]

border_coords_new = border_coords[::40]
ukraine_polygon = Polygon(border_coords_new)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.ops import nearest_points


# Function to filter points inside Ukraine
def filter_inside_ukraine(df):
    points = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    inside_mask = [ukraine_polygon.contains(p) for p in points]
    return df[inside_mask].reset_index(drop=True)

new_df['classification'] = new_df['classification'].astype(int)
# Apply filtering
new_df = filter_inside_ukraine(new_df)

# Get list of dates
dates = sorted(set(new_df['event_date']))
war_start_idx = 1510  # War starts at this index

# Assume full blue control before war
initial_grid = filter_inside_ukraine(pd.DataFrame({
    'longitude': np.linspace(new_df['longitude'].min(), new_df['longitude'].max(), 400),
    'latitude': np.linspace(new_df['latitude'].min(), new_df['latitude'].max(), 400),
    'probability': 1.0  # Entire Ukraine starts as blue
}))

class FrontlineNN(nn.Module):
    def __init__(self):
        super(FrontlineNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
# Generate a grid for predictions
def create_filtered_grid(df):
    # Use fixed bounds for consistent grid generation
    lon_min, lon_max = 22.0, 40.0  # Example bounds for Ukraine
    lat_min, lat_max = 44.0, 52.0
    
    lon_grid = np.linspace(lon_min, lon_max, 400)
    lat_grid = np.linspace(lat_min, lat_max, 400)
    
    grid_points = pd.DataFrame({
        'longitude': np.repeat(lon_grid, len(lat_grid)),
        'latitude': np.tile(lat_grid, len(lon_grid))})
    
    return filter_inside_ukraine(grid_points)


# Function to train and predict
def train_and_predict(current_df):
    X = current_df[['longitude', 'latitude']].values
    y = current_df['classification'].values.astype(np.float32)

    # Normalize input features
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / X_std

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Initialize neural network
    model = FrontlineNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    epochs = 500
    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    filtered_grid = create_filtered_grid(current_df)
    X_test = filtered_grid[['longitude', 'latitude']].values
    X_test_normalized = (X_test - X_mean) / X_std
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)

    # Predict probabilities
    with torch.no_grad():
        probabilities = model(X_test_tensor).numpy().reshape(-1)

    # Convert back to grid
    filtered_grid['probability'] = probabilities
    return filtered_grid

# Function to plot frontline
def plot_frontline(filtered_grid, current_df, title):
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([filtered_grid['longitude'].min(), filtered_grid['longitude'].max(),
                   filtered_grid['latitude'].min(), filtered_grid['latitude'].max()])
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)

    # Scatter event points
    red_points = current_df[current_df['classification'] == 1]
    blue_points = current_df[current_df['classification'] == 0]
    plt.scatter(red_points['longitude'], red_points['latitude'], color='red', label='Red Territory Events', s=10)
    plt.scatter(blue_points['longitude'], blue_points['latitude'], color='blue', label='Blue Territory Events', s=10)

    # Plot frontline (contour at 50% probability)
    lon_grid = filtered_grid.pivot_table(index='latitude', columns='longitude', values='probability')
    latitudes = lon_grid.index.values
    longitudes = lon_grid.columns.values
    probabilities = lon_grid.values

    plt.contour(longitudes, latitudes, probabilities, levels=[0.5], colors='black', linewidths=2)
    plt.legend()
    plt.title(title)
    plt.show()

# Track changes in control
red_gains = []
blue_gains = []

def track_changes(prev_grid, new_grid):
    global red_gains, blue_gains
    
    # Reset indexes to ensure alignment
    prev_grid = prev_grid.reset_index(drop=True)
    new_grid = new_grid.reset_index(drop=True)
    
    # Ensure both grids have the same points (same coordinates)
    # We'll merge them to align the points
    merged = pd.merge(prev_grid, new_grid, on=['longitude', 'latitude'], suffixes=('_prev', '_new'))
    
    # Count changes
    red_to_blue = ((merged['probability_prev'] > 0.5) & (merged['probability_new'] <= 0.5)).sum()
    blue_to_red = ((merged['probability_prev'] <= 0.5) & (merged['probability_new'] > 0.5)).sum()

    red_gains.append(blue_to_red)
    blue_gains.append(red_to_blue)  # Positive since blue gained

    print(f"Iteration: Red gained {blue_to_red}, Blue gained {red_to_blue}")

# Function to update frontline over time
def update_frontline():
    global initial_grid
    batch_size = 30

    # Create consistent grid first
    full_grid = create_filtered_grid(new_df)
    prev_grid = full_grid.copy()
    prev_grid['probability'] = 1.0  # Initialize all as blue
    
    all_events = pd.DataFrame(columns=new_df.columns)

    for i in range(war_start_idx, len(dates), batch_size):
        print(i)
        batch_dates = dates[i:i + batch_size]
        new_events = new_df[new_df['event_date'].isin(batch_dates)]
        
        # Add new events
        all_events = pd.concat([all_events, new_events], ignore_index=True)
        
        # Train and predict using consistent grid
        filtered_grid = train_and_predict(all_events)
        
        # Align grids - ensure same points
        merged = pd.merge(
            prev_grid[['longitude', 'latitude', 'probability']],
            filtered_grid[['longitude', 'latitude', 'probability']],
            on=['longitude', 'latitude'],
            suffixes=('_prev', '_new')
        )
        
        # Find changed points
        moved_mask = (merged['probability_prev'] > 0.5) != (merged['probability_new'] > 0.5)
        moved_coords = merged[moved_mask][['longitude', 'latitude']]
        
        if not moved_coords.empty:
            # Buffer moved points to define changing regions
            change_buffers = [Point(lon, lat).buffer(0.1)  # Buffer size in degrees (~5 km)
                              for lon, lat in zip(moved_coords['longitude'], moved_coords['latitude'])]
            changed_zone = unary_union(change_buffers)
        
            # Keep only old events that are NOT in the changed zone
            old_events_outside_change = all_events[~all_events.apply(
                lambda row: changed_zone.contains(Point(row['longitude'], row['latitude'])),
                axis=1
            )]
        
            # Keep new events as-is, and add them back
            all_events = pd.concat([old_events_outside_change, new_events], ignore_index=True)
        else:
            # If no changes in frontline, simply add new events
            all_events = pd.concat([all_events, new_events], ignore_index=True)
        
        # Calculate red-controlled area percentage (prob > 0.5 means red)
        red_area = (filtered_grid['probability'] > 0.5).sum()
        total_area = len(filtered_grid)
        red_percentage = (red_area / total_area) * 100
        area_control_percentages.append(red_percentage)
        # Update for next iteration
        prev_grid = filtered_grid.copy()

    plot_frontline(prev_grid, all_events, "Final Frontline")

area_control_percentages = []
update_frontline()

def plot_area_control():
    plt.figure(figsize=(10, 4))
    plt.plot(area_control_percentages, color='darkred', label='Red-Controlled Area %')
    plt.axhline(50, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Iteration (every 30 days)")
    plt.ylabel("Red-Controlled Area (%)")
    plt.title("Change in Territorial Control Over Time")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_area_control()