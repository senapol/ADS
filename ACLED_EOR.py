# RUN THIS FILE FOR HTML OR DOWNLOAD AND RUN THE HTML FILE
# this file combines Eyes on Russia and ACLED data for a more robust dataset

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.spatial import KDTree
import numpy as np

from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import savgol_filter
import numpy as np

def create_combined_events(x):
    acled_df = pd.read_csv('data/ACLED_Ukraine_Reduced.csv')
    eor_df = pd.read_csv('data/events.csv')

    # standardising
    acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])
    eor_df['date'] = pd.to_datetime(eor_df['date'])
    # eor_df['categories'] = eor_df['categories'].replace("'")
    print('standardised dates')

    acled_df['source_dataset'] = 'ACLED'
    eor_df['source_dataset'] = 'EOR'

    #EOR TO ACLED EVENT TYPE MAPPING
    eor_to_acled_mapping = {
        'Ground battle': {'event_type': 'Battles', 'sub_event_type': 'Armed clash'},
        'Russian Military Losses': {'event_type': 'Battles', 'sub_event_type': 'Armed clash'},
        'Ukrainian miitary losses': {'event_type': 'Battles', 'sub_event_type': 'Armed clash'},
        'Russian allies movements or losses': {'event_type': 'Battles', 'sub_event_type': 'Armed clash'},
        'Russian Firing Positions': {'event_type': 'Battles', 'sub_event_type': 'Armed clash'},
        'Russian Military Presence': {'event_type': 'Stategic developments', 'sub_event_type': 'Headquarters or base established'},
        'Bombing or explosion': {'event_type': 'Explosions/Remote violence', 'sub_event_type': 'Shelling/artillery/missile attack'},
        'Minitions': {'event_type': 'Explosions/Remote violence', 'sub_event_type': 'Shelling/artillery/missile attack'},
        'Civilian Casualty': {'event_type': 'Violence against civilians', 'sub_event_type': 'Attack'},
        'Civilian Infrastructure Damage' : {'event_type': 'Violence against civilians', 'sub_event_type': 'Attack'},
        'Environmental harm': {'event_type': 'Strategic developments', 'sub_event_type': 'Looting/property destruction'},
        'Mass grave or Burial': {'event_type': 'Violence against civilians', 'sub_event_type': 'Attack'},
        'Protest': {'event_type': 'Protests', 'sub_event_type': 'Peaceful protest'},
        'Detention or Arrest': {'event_type': 'Strategic developments', 'sub_event_type': 'Arrests'},
        'Military Infrastructure Damage': {'event_type': 'Strategic developments', 'sub_event_type': 'Looting/property destruction'},
        'Propaganda': {'event_type': 'Strategic developments', 'sub_event_type': 'Other'},
        'Other': {'event_type': 'Strategic developments', 'sub_event_type': 'Other'}
    }
    

    standard_columns = [
        'date',
        'latitude', 'longitude',
        'event_type',
        'sub_event_type',
        'location',
        'admin1',
        'description',
        'source_dataset',
        'source_id'
    ]

    acled_standard = pd.DataFrame({
        'date': acled_df['event_date'],
        'latitude': acled_df['latitude'],
        'longitude': acled_df['longitude'],
        'event_type': acled_df['event_type'],
        'sub_event_type': acled_df['sub_event_type'],
        'location': acled_df['location'],
        'admin1': acled_df['admin1'],
        'description': acled_df['notes'],
        'source_dataset': acled_df['source_dataset'],
        'source_id': acled_df['event_id_cnty']
    })

    eor_standard = pd.DataFrame({
        'date': eor_df['date'],
        'latitude': eor_df['latitude'],
        'longitude': eor_df['longitude'],
        'event_type': eor_to_acled_mapping[eor_df['categories']]['event_type'],
        'sub_event_type': eor_to_acled_mapping[eor_df['subcategories']]['sub_event_type'],
        'location': eor_df['city'],
        'admin1': eor_df['province'],
        'description': eor_df['description'],
        'source_dataset': eor_df['source_dataset'],
        'source_id': eor_df['id']
    })

    combined_df = pd.concat([acled_standard, eor_standard], ignore_index=True)
    combined_df = combined_df.sort_values('date')

    print(combined_df.head())
    # filtering only 2020 onwards:
    combined_df = combined_df[combined_df['date'] >= '2020-01-01']
    print(combined_df.head())

    combined_df.to_csv(f'data/combined_events{str(x)}.csv', index=False)

def calculate_event_density(df, date, radius=20):
    print('Calculating event density')

    prev_events = df[df['date'] <= date].copy()

    coords = prev_events[['latitude', 'longitude']].values
    print('Calculating KDTree')
    tree = KDTree(coords)
    print('Calculating density')
    density = []
    for i, row in prev_events.iterrows():
        point = np.array([row['latitude'], row['longitude']])
        print(f'Calculating density for point {point}')

        neighbours = tree.query_ball_point(point, radius/111.0) # 1 degree is approx 111 km
        density.append(len(neighbours))
    
    return density

def extract_frontline_density(df, date, window=60):
    # Filter for events in time window
    start_date = date - pd.Timedelta(days=window)
    events = df[(df['date'] <= date) & (df['date'] >= start_date)].copy()
    
    # Create a grid over Ukraine
    lon_min, lon_max = events['longitude'].min() - 0.5, events['longitude'].max() + 0.5
    lat_min, lat_max = events['latitude'].min() - 0.5, events['latitude'].max() + 0.5
    
    grid_size = 100
    lon_grid, lat_grid = np.mgrid[lon_min:lon_max:grid_size*1j, lat_min:lat_max:grid_size*1j]
    positions = np.vstack([lon_grid.ravel(), lat_grid.ravel()])
    
    # Calculate density using Gaussian KDE
    from scipy.stats import gaussian_kde
    points = events[['longitude', 'latitude']].values.T
    kde = gaussian_kde(points)
    density = kde(positions).reshape(grid_size, grid_size)
    
    # Extract contour at a specific density threshold
    from skimage import measure
    contours = measure.find_contours(density, 0.3 * density.max())
    
    # Convert contour indices to geographic coordinates
    frontlines = []
    for contour in contours:
        lon_scaled = lon_min + contour[:, 1] * (lon_max - lon_min) / grid_size
        lat_scaled = lat_min + contour[:, 0] * (lat_max - lat_min) / grid_size
        frontlines.append(np.column_stack([lon_scaled, lat_scaled]))
    
    return frontlines

def model():
    print('Creating model..')

    df = pd.read_csv('data/combined_events.csv')

    try:
        print(df.head())
    except:
        print('Error: Could not read file')
        exit()

    df['date'] = pd.to_datetime(df['date'])
    # df = df.sort_values('date', inplace=True)
    # df.reset_index(drop=True, inplace=True)

    # generate temporal features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.isocalendar().week
    df['day_of_week'] = df['date'].dt.dayofweek # Mon-Sun: 0-6

    invasion_date = pd.to_datetime('2022-02-24')
    df['days_since_invasion'] = (df['date'] - invasion_date).dt.days

    df['event_density'] = calculate_event_density(df, df['date'].max())


    print('data in time series:')
    print(df.sample(10))

def print_event_types():
    # create_combined_events(2)
    df = pd.read_csv('data/combined_events.csv')

    df_eor = df[df['source_dataset'] == 'EOR']
    # df_eor['date'] = pd.to_datetime(df_eor['date'])
    eor_types = []
    for event in df_eor['event_type']:
        i = 0
        for e in event.replace("'", "").replace('"', '').replace(']', '').replace('[', '').split(','):
            i += 1
            if i > 1:
                print('multiple event types:', event)
            if e not in eor_types:
                eor_types.append(e.strip())
    eor_types.sort()
    print('EOR event types:', set(eor_types))

    df_acled = df[df['source_dataset'] == 'ACLED']
    # df_acled['date'] = pd.to_datetime(df_acled['date'])
    acled_types = df_acled['event_type'].unique()
    print('ACLED event types:', set(acled_types))

    #min max dates@
    print('Max/min dates:\n----------------------------')
    print('ACLED min date:', df_acled['date'].min())
    print('ACLED max date:', df_acled['date'].max())

    print('EOR min date:', df_eor['date'].min())
    print('EOR max date:', df_eor['date'].max())
    print('Combined min date:', df['date'].min())
    print('Combined max date:', df['date'].max())
    print('----------------------------')
    print('ACLED count:', len(df_acled))
    print('EOR count:', len(df_eor))
    print('----------------------------')
    #sum per event type
    print('ACLED event type counts:')
    print(df_acled['event_type'].value_counts())
    print('----------------------------')
    print('EOR event type counts:')
    print(df_eor['event_type'].value_counts())

print_event_types()