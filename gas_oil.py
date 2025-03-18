# integrating AGSI and oil data
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

API_KEY = os.getenv("API_KEY")

base_url = "https://agsi.gie.eu/api"

headers = {
    "x-key": API_KEY
}

def get_country_data(country_code, start_date, end_date):
    params = {
        "country": country_code,
        "from": start_date,
        "to": end_date,
        "size": 1000 
    }
    try:
        response = requests.get(base_url, headers=headers, params=params)
    except Exception as e:
        print(f"Error: {e}")

    if response.status_code == 200:
        data = response.json()
        # Convert to DataFrame for analysis
        df = pd.DataFrame(data['data'])
        # print(df.head())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

    return data

def save_AGSI_data():
    response = requests.get(base_url, headers=headers)
    countries = []
    if response.status_code == 200:
        data = response.json()
        
        for country in data['data'][0]['children']:
            countries.append(country['code'])

        print(f'EU countries: {countries}')

        for country in data['data'][1]['children']:
            countries.append(country['code'])

        print(f'Non-EU countries: {countries}')

    start_date = "2022-01-01"
    end_date = "2022-12-31"

    key_countries = ['DE', 'FR', 'PL', 'UA']
    all_data = {}

    for country in key_countries:
        print(f"Getting data for {country}")
        country_data = get_country_data(country, start_date, end_date)

        if country_data:
            df = pd.DataFrame(country_data['data'])

            print(f"Data for {country}:")
            print(df.head())
            print(df.describe())
            print(f'Columns: {df.columns}')

            df['gasDayStart'] = pd.to_datetime(df['gasDayStart'])

            for col in ['gasInStorage', 'consumption', 'consumptionFull', 'injection', 'withdrawal', 'workingGasVolume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.sort_values('gasDayStart')

            all_data[country] = df

            print(f"Data for {country} loaded successfully")

    for country, df in all_data.items():
        df.to_csv(f'data/agsi_data_{country}.csv', index=False)

def eia_oil():

    df = pd.read_excel('data/eia_oil.xls', sheet_name='Data 1', skiprows=2)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    df['Europe Brent Spot Price FOB (Dollars per Barrel)'] = pd.to_numeric(df['Europe Brent Spot Price FOB (Dollars per Barrel)'], errors='coerce')
    df['brent_price_usd'] = df['Europe Brent Spot Price FOB (Dollars per Barrel)']
    df = df.drop(columns=['Europe Brent Spot Price FOB (Dollars per Barrel)'])
    df = df[df['Date'] >= '2020-01-01']
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    print(df.isna().sum())

    # making line plot
    # fig = px.line(df, x='Date', y='brent_price_usd', title='Europe Brent Spot Price FOB (Dollars per Barrel)')
    # fig.show()

    key_events = {
        '2022-02-24': 'Russia invades Ukraine',
        '2022-03-08': 'US and UK ban Russian oil imports',
        '2022-06-03': 'EU 6th Sanctions Package'
    }

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['brent_price_usd'])

    # Add event markers
    for date, label in key_events.items():
        event_date = pd.to_datetime(date)
        plt.axvline(x=event_date, color='r', linestyle='--', alpha=0.7)
        price_at_event = df.loc[df['Date'] >= event_date, 'brent_price_usd'].iloc[0]
        plt.text(event_date, 0 + 5, label, rotation=90)

    plt.title('Brent Crude Oil Prices (2020-Present)')
    plt.ylabel('USD per Barrel')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
        # plt.savefig('brent_oil_timeline.png')
    plt.show()

def integrate():
    df_de = pd.read_csv('data/agsi_data_DE.csv')
    df_fr = pd.read_csv('data/agsi_data_FR.csv')
    df_pl = pd.read_csv('data/agsi_data_PL.csv')
    df_ua = pd.read_csv('data/agsi_data_UA.csv')
    country_dfs = {}
    numeric_cols = ['gasInStorage','consumption','consumptionFull','injection','withdrawal','netWithdrawal','workingGasVolume','injectionCapacity','withdrawalCapacity']

    for df, code in zip([df_de, df_fr, df_pl, df_ua], ['DE', 'FR', 'PL', 'UA']):
        df_processed = df.copy()
        df_processed['gasDayStart'] = pd.to_datetime(df_processed['gasDayStart'])
        df_processed = df_processed[df_processed['gasDayStart'] >= '2020-01-01']
        for col in numeric_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        country_dfs[code] = df_processed
        print(f"{code}: {df_processed.shape}")

    df_oil = pd.read_excel('data/eia_oil.xls', sheet_name='Data 1', skiprows=2)
    df_oil['Date'] = pd.to_datetime(df_oil['Date'])
    df_oil['brent_price_usd'] = df_oil['Europe Brent Spot Price FOB (Dollars per Barrel)']
    df_oil = df_oil.drop(columns=['Europe Brent Spot Price FOB (Dollars per Barrel)'])

    print(country_dfs.keys())
    merged_dfs = {}

    for country, df in country_dfs.items():

        merged = pd.merge(
            df,
            df_oil,
            how='left',
            left_on='gasDayStart',
            right_on='Date'
        )

        merged_dfs[country] = merged
        print(f"{country}: {merged.shape}")

    for country, merged_df in merged_dfs.items():
        print(f"\n--- {country} Data Sample ---")
        print(merged_df.head(3))  # Show first 3 rows
        print(f"Country name: {merged_df['name'].unique()}")
        print(f"URL: {merged_df['url'].unique()}")

    merged_dfs.to_csv('data/merged_data.csv', index=False)
    # print(merged.head())
    # print(merged.describe())
    # print(merged['url'].unique())
    # print(merged['name'].unique())

eia_oil()