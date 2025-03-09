# dataset columns explained: https://acleddata.com/knowledge-base/codebook/#acled-data-columns-at-a-glance

import pandas as pd
import plotly.express as px

def reduce_acled():
    try:
        df = pd.read_csv('data/Ukraine_Black_Sea_2020_2025_Feb28.csv')
    except Exception as e:
        print(f'Error {e}: Could not read file')
        exit()
    # pd.set_option('display.max_columns', None)

    essential_columns = [
        'event_id_cnty',  # Unique identifier
        'event_date',     # Date of event
        'year',           # Year of event
        'event_type',     # Type of event
        'sub_event_type', # Subtype of event
        'actor1',         # Primary actor
        'actor2',         # Secondary actor
        'country',        # Country
        'admin1',         # Admin level 1 (province/oblast)
        'admin2',         # Admin level 2 (district)
        'location',       # Location name
        'latitude',       # Latitude
        'longitude',      # Longitude
        'fatalities',     # Fatalities
        'notes'           # Brief description
    ]

    # Filter to only include columns that exist in the dataset
    available_columns = [col for col in essential_columns if col in df.columns]
    df_reduced = df[available_columns].copy()

    df_reduced['event_date'] = pd.to_datetime(df_reduced['event_date'])

    output = 'data/ACLED_Ukraine_Reduced.csv'
    df_reduced.to_csv(output, index=False)
    print(f'File saved to {output}')


df = pd.read_csv('data/ACLED_Ukraine_Reduced.csv')

print(df.head())

print('\n[*] Dataset summary:')
print('---------------------')
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
print(f"Total events: {len(df)}")
print(f"Total fatalities: {df['fatalities'].sum()}")
print('---------------------')
missing = df.isnull().sum()
print(f"Missing values: \n{missing[missing > 0]}")
print('---------------------')
event_counts = df['event_type'].value_counts()
fatalities_by_event = df.groupby('event_type')['fatalities'].sum()
event_counts = pd.concat([event_counts, fatalities_by_event], axis=1)

print("Event types and fatalities:")
print(event_counts)

# average fatalities per event
event_counts['Avg_Fatalities'] = event_counts['fatalities'] / event_counts['count']

# bubble chart of event count vs fataliites by event type
fig = px.scatter(
    event_counts.reset_index(),
    x='count',
    y='fatalities',
    size='Avg_Fatalities',
    color='event_type', 
    hover_name='event_type',
    text='event_type',
    log_x=True,  # Use log scale if counts vary widely
    log_y=True,  # Use log scale if fatalities vary widely
    size_max=60
)

fig.update_layout(
    title="Event Counts vs. Fatalities by Event Type",
    xaxis_title="Number of Events (log scale)",
    yaxis_title="Total Fatalities (log scale)",
    height=600,
    width=900,
    showlegend=False
)

fig.update_traces(textposition='top center')

fig.show()