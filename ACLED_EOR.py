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

def create_combined_events():
    acled_df = pd.read_csv('data/ACLED_Ukraine_Reduced.csv')
    eor_df = pd.read_csv('data/events.csv')

    # standardising
    acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])
    eor_df['date'] = pd.to_datetime(eor_df['date'])
    print('standardised dates')

    acled_df['source_dataset'] = 'ACLED'
    eor_df['source_dataset'] = 'EOR'

    # EOR TO ACLED EVENT TYPE MAPPING
    # event_type_mapping = {
    #     'Russian Military Presence': 
    # }

    standard_columns = [
        'date',
        'latitude', 'longitude',
        'event_type',
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
        'event_type': eor_df['categories'],
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

    combined_df.to_csv('data/combined_events.csv', index=False)

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

def extract_frontline(df, date, method='dbscan'): # density based spatial clustering of applications with noise'''

    # https://hogonext.com/how-to-combine-clustering-with-regression-for-prediction/
    # https://www.jstor.org/stable/20441166
    print(f'Extracting frontline for {date}')
    
    events = df[df['date'] <= date].copy()
    coords = events[['latitude', 'longitude']].values

    print('Using DBSCAN for clustering')
    clustering = DBSCAN(eps=0.1, min_samples=5).fit(coords)

    events['cluster'] = clustering.labels_

    frontlines = {}

    for cluster_id in sorted(set(clustering.labels_)):
        if cluster_id == -1: # noise
            continue
        
        cluster_points = events[events['cluster'] == cluster_id]

        if len(cluster_points) < 10: # not enough points to fit a model
            continue

        print('Random forest regression model to extract front line')

        n_estimators = 100
        X = events['longitude'].values.reshape(-1,1)
        y = events['latitude'].values

        #train RFR model
        rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        rf_model.fit(X, y)

        x_dense = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)

        y_pred = rf_model.predict(x_dense)

        y_smooth = savgol_filter(y_pred, window_length=15, polyorder=3)

        frontline = np.column_stack((x_dense.flatten(), y_smooth))
        frontlines[date] = frontline

    if method == 'kde': # kernel density estimation
        pass

    return frontlines


def visualise_map():
    print('Loading combined data')

    df = pd.read_csv('data/combined_events.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df.reset_index(drop=True, inplace=True)
    df['month_year'] = df['date'].dt.strftime('%Y-%m')

    df_acled = df[df['source_dataset'] == 'ACLED']
    df_eor = df[df['source_dataset'] == 'EOR']

    fig = make_subplots()

    print('creating initial figure')

    fig_acled = px.scatter_mapbox(
        df_acled,
        lat='latitude',
        lon='longitude',
        color='event_type',
        hover_name='location',
        hover_data={
            'date': True,
            'event_type': True,
            'source_dataset': True,
            'description': True,
            'latitude': False,
            'longitude': False
        },
        animation_frame='month_year',
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 49.0, "lon": 31.0},
        height=800,
        width=1200,
        title='Ukraine Conflict Events (ACLED & Eyes on Russia)'
    )

    fig_eor = px.scatter_mapbox(
        df_eor,
        lat='latitude',
        lon='longitude',
        color='event_type',
        hover_name='location',
        hover_data={
            'date': True,
            'event_type': True,
            'source_dataset': True,
            'description': True,
            'latitude': False,
            'longitude': False
        },
        animation_frame='month_year',
        mapbox_style="carto-positron",
        zoom=5,
        center={"lat": 49.0, "lon": 31.0}
    )

    print('Adding traces to figure')
    for trace in fig_acled.data:
        trace.name = f"ACLED - {trace.name}" if hasattr(trace, 'name') else "ACLED"
        fig.add_trace(trace)

    for trace in fig_eor.data:
        trace.name = f"Eyes on Russia - {trace.name}" if hasattr(trace, 'name') else "EOR"
        fig.add_trace(trace)

    frames = []

    print('Calculating frontlines for each month')

    frontlines_by_month = {}

    sample_months=sorted(df['month_year'].unique())[::3] # every 3 months


    for month_year in sorted(df['month_year'].unique()):
        # get frames for this month from both figs
        acled_frame = next((f for f in fig_acled.frames if f.name == month_year), None)
        eor_frame = next((f for f in fig_eor.frames if f.name == month_year), None)
        
        if acled_frame and eor_frame:
            # combine the data from both frames
            combined_data = list(acled_frame.data) + list(eor_frame.data)
            frames.append(go.Frame(data=combined_data, name=month_year))

    fig.frames = frames

    # animation controls and layout
    fig.update_layout(
        title='Ukraine Conflict Events (ACLED & Eyes on Russia)',
        mapbox=dict(
            style="carto-positron",
            zoom=5,
            center={"lat": 49.0, "lon": 31.0}
        ),
        height=800,
        width=1200,
        
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }
        ],
        
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Date: ",
                "visible": True,
                "xanchor": "center"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [frame.name],
                        {"frame": {"duration": 300, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 300}}
                    ],
                    "label": frame.name,
                    "method": "animate"
                } for frame in fig.frames
            ]
        }],
        
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    )

    acled_indices = [i for i, trace in enumerate(fig.data) if 'ACLED' in trace.name]
    eor_indices = [i for i, trace in enumerate(fig.data) if 'EOR' in trace.name]

    print('sorting buttons')
    # visibility settings for each button
    all_visible = [True] * len(fig.data)
    acled_only = [i in acled_indices for i in range(len(fig.data))]
    eor_only = [i in eor_indices for i in range(len(fig.data))]

    # source filter buttons
    buttons = [
        dict(
            label="All Sources",
            method="update",
            args=[{"visible": all_visible}]
        ),
        dict(
            label="ACLED Only",
            method="update",
            args=[{"visible": acled_only}]
        ),
        dict(
            label="Eyes on Russia Only",
            method="update",
            args=[{"visible": eor_only}]
        ),
        dict(
            label="Reset View",
            method="relayout",
            args=[{"mapbox.center": {"lat": 49.0, "lon": 31.0}, "mapbox.zoom": 5}]
        )
    ]

    # add buttons to layout
    fig.update_layout(
        updatemenus=[
            # first updatemenus item is the animation control
            fig.layout.updatemenus[0],
            # add source filters as second updatemenus item
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.9,
                xanchor="right",
                y=0.99,
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.5)"
            )
        ]
    )

    print('saving to html')
    # save to HTML
    fig.write_html("Ukraine_Conflict_Map.html")

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


model()