# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 19:25:34 2025

@author: talia
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ipywidgets import interactive, IntSlider, fixed
from IPython.display import display
import datetime

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
    
    armed_clash_events = df[df['sub_event_type'] == 'Government regains territory'].copy()

    # Swap actor1 and actor2 in the duplicated rows
    armed_clash_events[['actor1', 'actor2']] = armed_clash_events[['actor2', 'actor1']]

    # Append duplicated rows back to the original DataFrame
    df = pd.concat([df, armed_clash_events], ignore_index=True)
    
    armed_clash_events = df[df['sub_event_type'] == 'Non-state actor overtakes territory'].copy()

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
date = dates[500:600]


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data (replace with actual file)
df = new_df  # Contains 'longitude', 'latitude', 'classification'

# Convert 'classification' into binary (0: red, 1: blue)
#df['classification'] = df['classification'].map({'red': 0, 'blue': 1})

# Extract spatial features
X = df[['longitude', 'latitude']].values  # Spatial features
y = df['classification'].values

# Start with initial time window
start_idx, end_idx = 1510, 1515
current_dates = dates[start_idx:end_idx]
current_df = df[df['event_date'].isin(current_dates)]

# Neural Network Model
class FrontlineNN(nn.Module):
    def __init__(self):
        super(FrontlineNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Train & Predict function
def train_and_predict(current_df):
    X = current_df[['longitude', 'latitude']].values
    y = current_df['classification'].values

    # Normalize coordinates
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_normalized = (X - X_mean) / X_std

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Train model
    model = FrontlineNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 500
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    # Generate prediction grid
    lon_min, lon_max = current_df['longitude'].min(), current_df['longitude'].max()
    lat_min, lat_max = current_df['latitude'].min(), current_df['latitude'].max()
    lon_grid, lat_grid = np.meshgrid(np.linspace(lon_min, lon_max, 400),
                                     np.linspace(lat_min, lat_max, 400))
    X_test = np.c_[lon_grid.ravel(), lat_grid.ravel()]
    X_test_normalized = (X_test - X_mean) / X_std
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)

    # Predict probabilities
    with torch.no_grad():
        probabilities = model(X_test_tensor).numpy().reshape(lon_grid.shape)

    return lon_grid, lat_grid, probabilities, model, X_mean, X_std

# Function to remove old points only if 4+ new points exist nearby
def filter_old_points(current_df, new_events):
    buffer_distance = 0.1  # Define removal radius
    cleaned_df = current_df.copy()

    for _, row in new_events.iterrows():
        lon, lat = row['longitude'], row['latitude']

        # Find new points in the same area
        nearby_new = new_events[(np.abs(new_events['longitude'] - lon) < buffer_distance) &
                                (np.abs(new_events['latitude'] - lat) < buffer_distance)]
        
        # If 4 or more new points, remove older points in the area
        if len(nearby_new) >= 4:
            cleaned_df = cleaned_df[~((np.abs(cleaned_df['longitude'] - lon) < buffer_distance) &
                                      (np.abs(cleaned_df['latitude'] - lat) < buffer_distance))]
    
    return cleaned_df

# Function to update frontline dynamically
def update_frontline():
    global current_dates, current_df

    print("Starting frontline update...")

    # Initial Frontline Plot
    print("Plotting initial frontline...")
    lon_grid, lat_grid, probabilities, _, _, _ = train_and_predict(current_df)
    plot_frontline(lon_grid, lat_grid, probabilities, current_df, title="Initial Frontline")

    # Process dates in batches of 30 days
    batch_size = 30
    for i in range(end_idx, len(dates), batch_size):
        batch_dates = dates[i:i + batch_size]
        print(f"Processing batch: {batch_dates[0]} to {batch_dates[-1]}")

        # Add new events
        new_events = df[df['event_date'].isin(batch_dates)]
        current_df = pd.concat([current_df, new_events])

        # Train model and predict
        lon_grid, lat_grid, probabilities, model, X_mean, X_std = train_and_predict(current_df)

        # Test model performance on new events
        X_new = new_events[['longitude', 'latitude']].values
        X_new_normalized = (X_new - X_mean) / X_std
        X_new_tensor = torch.tensor(X_new_normalized, dtype=torch.float32)

        with torch.no_grad():
            new_predictions = model(X_new_tensor).numpy()

        # Identify misclassified points
        misclassified = np.abs(new_predictions - new_events['classification'].values.reshape(-1, 1)) > 0.5
        if np.any(misclassified):
            print("Misclassification detected! Adjusting frontline data.")

            # Remove old points in affected areas only if enough new points exist
            current_df = filter_old_points(current_df, new_events)

        # Track gains/losses
        track_changes(probabilities, lon_grid, lat_grid)

    # Final Frontline Plot
    print("Plotting final frontline...")
    lon_grid, lat_grid, probabilities, _, _, _ = train_and_predict(current_df)
    plot_frontline(lon_grid, lat_grid, probabilities, current_df, title="Final Frontline")

# Function to plot the frontline
def plot_frontline(lon_grid, lat_grid, probabilities, current_df, title):
    fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    # Overlay probability map
    contour = ax.contourf(lon_grid, lat_grid, probabilities, levels=20, cmap="coolwarm", alpha=0.5, transform=ccrs.PlateCarree())
    plt.colorbar(contour, ax=ax, label="Probability of 'blue'")

    # Scatter plot of events
    ax.scatter(current_df['longitude'], current_df['latitude'], c=current_df['classification'], cmap="bwr", edgecolors="k", transform=ccrs.PlateCarree())

    # Plot frontline (50% decision boundary)
    ax.contour(lon_grid, lat_grid, probabilities, levels=[0.5], colors="black", linewidths=2, transform=ccrs.PlateCarree())

    plt.title(title)
    plt.show()

# Function to track changes over time
def track_changes(probabilities, lon_grid, lat_grid):
    global previous_probabilities

    if 'previous_probabilities' in globals():
        frontline_change = probabilities - previous_probabilities

        # Identify where changes occur
        red_gains = np.sum((frontline_change > 0.5).astype(int))
        blue_gains = np.sum((frontline_change < -0.5).astype(int))

        if red_gains > blue_gains:
            print("🔴 Red is advancing!")
        elif blue_gains > red_gains:
            print("🔵 Blue is advancing!")
        else:
            print("No significant change in the frontline.")

    # Store current probabilities for next iteration
    previous_probabilities = probabilities.copy()

# Run the dynamic frontline update
update_frontline()