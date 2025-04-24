# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 12:32:31 2025

@author: talia
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point

# Load event data
df = pd.read_csv("data/combined_events.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Filter only 'Battles' events
df = df[df["event_type"] == "Battles"].copy()

# Sort by date
df = df.sort_values(by="date")

# Determine event groupings based on equal event distribution
n_groups = 60  # Number of desired event groups
df["group"] = pd.qcut(df.index, n_groups, labels=False)

time_slices = df["group"].unique()

# Get date ranges for each group
group_dates = df.groupby("group")["date"].agg(["min", "max"])

# Initialize parameters
n_particles = 1000
n_iterations = 10

def run_particle_filter(latitudes, longitudes):
    particles = np.column_stack((
        np.random.uniform(min(latitudes), max(latitudes), n_particles),
        np.random.uniform(min(longitudes), max(longitudes), n_particles)
    ))
    weights = np.ones(n_particles) / n_particles
    
    for _ in range(n_iterations):
        for i in range(len(particles)):
            distances = np.sqrt((latitudes - particles[i, 0])**2 + (longitudes - particles[i, 1])**2)
            weights[i] = np.exp(-np.min(distances))
        
        weights += 1e-10  # Avoid zero weights
        weights /= np.sum(weights)  # Normalize
        
        indices = np.random.choice(range(len(particles)), size=len(particles), p=weights)
        particles = particles[indices]
    
    return particles

# Create color gradient
cmap = cm.get_cmap("coolwarm", len(time_slices))
norm = mcolors.Normalize(vmin=0, vmax=len(time_slices))

fig, ax = plt.subplots(figsize=(8, 6))
front_line_centers = []
# Load Natural Earth data correctly
try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
except AttributeError:
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
world.plot(ax=ax, color="lightgrey")
for i, group in enumerate(time_slices):
    subset = df[df["group"] == group]
    latitudes, longitudes = subset["latitude"].values, subset["longitude"].values
    
    if len(latitudes) == 0:
        continue
    
    particles = run_particle_filter(latitudes, longitudes)
    center_lat = np.mean(particles[:, 0])
    center_lon = np.mean(particles[:, 1])
    front_line_centers.append((center_lon, center_lat))
    date_range = f"{group_dates.loc[group, 'min'].date()} - {group_dates.loc[group, 'max'].date()}"
    ax.scatter(particles[:, 1], particles[:, 0], s=5, color=cmap(norm(i)), label=date_range if i % 2 == 0 else "")
    ax.set_xlim(min(longitudes) - 2, max(longitudes) + 2)  # Zooming in on longitude range
    ax.set_ylim(min(latitudes) - 2, max(latitudes) + 2)  # Zooming in on latitude range
    
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Front Line Progression Over Time (Battles Only)")
ax.legend(title="Date Range", loc="upper right", fontsize="small")
plt.show()

# Plot trajectory of front line movement with map overlay
fig, ax = plt.subplots(figsize=(8, 6))

# Load Natural Earth data correctly
try:
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
except AttributeError:
    world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

world.plot(ax=ax, color="lightgrey")
lons, lats = zip(*front_line_centers)
ax.plot(lons, lats, marker="o", linestyle="-", color="blue", label="Front Line Movement")
ax.set_xlim(min(lons) - 0.5, max(lons) + 0.5)
ax.set_ylim(min(lats) - 0.5, max(lats) + 0.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Front Line Movement Over Time with Map Overlay")
ax.legend()
plt.show()

# Compute displacement of the front line
displacements = []
for i in range(1, len(front_line_centers)):
    displacement = geodesic(front_line_centers[i-1][::-1], front_line_centers[i][::-1]).km
    displacements.append(displacement)

# Plot displacement over time
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(group_dates["max"].iloc[1:], displacements, marker="o", linestyle="-", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Displacement (km)")
ax.set_title("Displacement of the Front Line Over Time")
plt.xticks(rotation=45)
plt.show()
