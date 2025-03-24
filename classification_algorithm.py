# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 11:32:37 2025

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
date = dates[1550:1555]
#print(new_df)
plot_classification_scatter(new_df[new_df['event_date'].isin(date)])
new_df_2 = new_df[new_df['event_date'].isin(date)]


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data (replace with actual file)
df = new_df_2  # Contains 'longitude', 'latitude', 'classification'

# Convert 'classification' into binary (0: red, 1: blue)
#df['classification'] = df['classification'].map({'red': 0, 'blue': 1})

# Extract spatial features
X = df[['longitude', 'latitude']].values  # Spatial features
y = df['classification'].values

# Normalize input features (longitude and latitude)
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_normalized = (X - X_mean) / X_std

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Make y a column vector

# Define Neural Network
class FrontlineNN(nn.Module):
    def __init__(self):
        super(FrontlineNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output probability (0-1)
        )

    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = FrontlineNN()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Generate a grid for prediction
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_grid, lat_grid = np.meshgrid(np.linspace(lon_min, lon_max, 400),
                                 np.linspace(lat_min, lat_max, 400))
X_test = np.c_[lon_grid.ravel(), lat_grid.ravel()]
X_test_normalized = (X_test - X_mean) / X_std  # Apply same normalization
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)

# Predict probabilities
with torch.no_grad():
    probabilities = model(X_test_tensor).numpy().reshape(lon_grid.shape)

# Plot with Cartopy
fig, ax = plt.subplots(figsize=(12, 7), subplot_kw={'projection': ccrs.PlateCarree()})

# Add map features
ax.set_extent([lon_min - 1, lon_max + 1, lat_min - 1, lat_max + 1], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# Overlay probability map
contour = ax.contourf(lon_grid, lat_grid, probabilities, levels=20, cmap="coolwarm", alpha=0.5, transform=ccrs.PlateCarree())
plt.colorbar(contour, ax=ax, label="Probability of 'blue'")

# Overlay event points
ax.scatter(df['longitude'], df['latitude'], c=y, cmap="bwr", edgecolors="k", transform=ccrs.PlateCarree(), label="Events")

# Plot the frontline (50% decision boundary)
ax.contour(lon_grid, lat_grid, probabilities, levels=[0.5], colors="black", linewidths=2, transform=ccrs.PlateCarree())

# Labels & Title
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Front Line Prediction using PyTorch Neural Network with Cartopy")
ax.legend()


# =============================================================================
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from scipy.spatial import KDTree
# from scipy.stats import gaussian_kde
# 
# # Load data (assume df contains event data)
# # Columns: ['longitude', 'latitude', 'classification']
# df = new_df_2  # Replace with actual data
# 
# # Extract relevant features
# X = df[['longitude', 'latitude']].values  # Spatial features
# y = df['classification'].values
# 
# # Calculate the proportion of red to blue points
# red_count = np.sum(y == 0)  # Number of red points
# blue_count = np.sum(y == 1)  # Number of blue points
# total_count = len(y)  # Total number of points
# 
# # Calculate weights inversely proportional to class frequencies
# weight_red = total_count / (2 * red_count)  # Weight for red points
# weight_blue = total_count / (2 * blue_count)  # Weight for blue points
# 
# # Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# 
# # Train an SVM with RBF kernel and dynamically calculated class weights
# svm_model = SVC(kernel='rbf', C=1.0, probability=True, class_weight={0: 1/weight_red, 1: 1/weight_blue})
# svm_model.fit(X_scaled, y)
# 
# # Generate a grid for decision boundary
# lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
# lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
# lon_grid, lat_grid = np.meshgrid(np.linspace(lon_min, lon_max, 200),
#                                  np.linspace(lat_min, lat_max, 200))
# X_test = np.c_[lon_grid.ravel(), lat_grid.ravel()]
# X_test_scaled = scaler.transform(X_test)
# 
# # Predict class probabilities
# probabilities = svm_model.predict_proba(X_test_scaled)[:, 1]  # Probability of "blue"
# probabilities = probabilities.reshape(lon_grid.shape)
# 
# # Adjust prediction: If both 'red' and 'blue' are nearby, make it contested
# tree_red = KDTree(X[y == 0])  # Locations of 'red' events
# tree_blue = KDTree(X[y == 1])  # Locations of 'blue' events
# 
# # Compute distance to nearest red and blue event for each grid point
# dist_to_red, _ = tree_red.query(X_test)
# dist_to_blue, _ = tree_blue.query(X_test)
# 
# # Define a contested area threshold dynamically based on point density
# contested_threshold = 0.00005 * (lon_max - lon_min)  # Reduced threshold
# 
# # Mark areas as contested where both red and blue events are nearby
# contested_mask = (dist_to_red < contested_threshold) & (dist_to_blue < contested_threshold)
# 
# # Set probabilities to 0.5 in contested areas (indicating a front line)
# probabilities_flat = probabilities.ravel()
# probabilities_flat[contested_mask] = 0.5  # Emphasize contested areas
# probabilities = probabilities_flat.reshape(lon_grid.shape)
# 
# # Plot the decision boundary with contested zones on a geographical map
# plt.figure(figsize=(12, 8))
# 
# # Create a Cartopy plot with PlateCarree projection
# ax = plt.axes(projection=ccrs.PlateCarree())
# 
# # Add geographical features (country borders, coastlines, etc.)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black')
# ax.add_feature(cfeature.OCEAN)
# 
# # Plot the decision boundary
# contour = ax.contourf(lon_grid, lat_grid, probabilities, levels=20, cmap="coolwarm", alpha=0.7, transform=ccrs.PlateCarree())
# plt.colorbar(contour, label="Probability of 'blue'")
# 
# # Plot the events
# scatter = ax.scatter(df['longitude'], df['latitude'], c=y, cmap="bwr", edgecolors="k", label="Events", transform=ccrs.PlateCarree())
# 
# # Highlight the contested boundary with a thicker black line
# ax.contour(lon_grid, lat_grid, probabilities, levels=[0.5], colors="black", linewidths=2, linestyles="dashed", transform=ccrs.PlateCarree())
# 
# # Add gridlines and labels
# ax.gridlines(draw_labels=True)
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.set_title("Estimated Front Line with Contested Zones on Geographical Map")
# 
# # Add a legend
# plt.legend()
# 
# # Show the plot
# plt.show()
# =============================================================================
