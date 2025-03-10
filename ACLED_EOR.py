# RUN THIS FILE FOR HTML OR DOWNLOAD AND RUN THE HTML FILE
# this file combines Eyes on Russia and ACLED data for a more robust dataset

import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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

print('Loading combined data')

df = pd.read_csv('data/combined_events.csv')
df['date'] = pd.to_datetime(df['date'])
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
frames = []
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
