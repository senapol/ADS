import pandas as pd

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


