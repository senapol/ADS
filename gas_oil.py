# integrating AGSI and oil data
import requests
import pandas as pd
from datetime import datetime as dt
import json
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px

API_KEY = '7bc824f059b3f2d877b290f6f37ce789'
print(API_KEY)

base_url = "https://agsi.gie.eu/api"

headers = {
    "x-key": API_KEY
}

def get_country_data(country_code, start_date, end_date):
    all_data = []
    current_start = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    while current_start < end_date:
        current_end = min(current_start + pd.Timedelta(days=250), end_date)

        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        print(f"Fetching data for {country_code} from {start_str} to {end_str}")
        params = {
            "country": country_code,
            "from": start_str,
            "to": end_str,
            "size": 250 
        }

        try:
            response = requests.get(base_url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    chunk_df = pd.DataFrame(data['data'])
                    all_data.append(chunk_df)
                    print(f'Fetched {len(chunk_df)} records')
                else:
                    print(f"No data for {country_code} from {start_str} to {end_str}")
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error: {e}")
        
        current_start = current_end + pd.Timedelta(days=1)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['gasDayStart'] = pd.to_datetime(combined_df['gasDayStart'])

        for col in ['gasInStorage', 'consumption', 'consumptionFull', 'injection', 'withdrawal', 'workingGasVolume']:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

        combined_df = combined_df.sort_values('gasDayStart')
        combined_df = combined_df.drop_duplicates(subset=['gasDayStart'])
        print(f'Total records for {country_code}: {len(combined_df)}')
        return combined_df

    return None

def save_AGSI_data():
    response = requests.get(base_url, headers=headers)
    countries = []
    if response.status_code == 200:
        data = response.json()
        # print(data.keys())
        # print(data)
        
        for country in data['data'][0]['children']:
            countries.append(country['code'])

        print(f'EU countries: {countries}')

        for country in data['data'][1]['children']:
            countries.append(country['code'])

        print(f'Non-EU countries: {countries}')

    start_date = "2020-01-01"
    end_date = "2025-03-13"

    key_countries = ['DE', 'FR', 'PL', 'UA']

    for country in key_countries:
        print(f"Getting data for {country}")
        country_df = get_country_data(country, start_date, end_date)

        if country_df is not None:
            output_file = f'data/agsi_data_{country}_complete.csv'
            country_df.to_csv(output_file, index=False) 
            print(f'Saved data for {country} to {output_file}')
        else:
            print(f"No data found for {country}")

    # for country, df in all_data.items():
        # df.to_csv(f'data/agsi_data_{country}.csv', index=False, line_terminator='\n')

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
    df_de = pd.read_csv('data/agsi_data_DE_complete.csv')
    df_fr = pd.read_csv('data/agsi_data_FR_complete.csv')
    df_pl = pd.read_csv('data/agsi_data_PL_complete.csv')
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
    df_oil['Date'] = df_oil['Date'].ffill
    df_oil['brent_price_usd'] = df_oil['brent_price_usd'].ffill()
 

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

    combined_countries_df = pd.concat(merged_dfs.values(), ignore_index=True)
    #random sample
    print(combined_countries_df.sample(5))
    print(combined_countries_df.describe())
    null_values = combined_countries_df[combined_countries_df.isnull().any(axis=1)]
    
    combined_countries_df.to_csv('data/oil_combined_countries.csv', index=False)
    # print(merged.head())
    # print(merged.describe())
    # print(merged['url'].unique())
    # print(merged['name'].unique())

def graphs():
    df = pd.read_csv('data/oil_combined_countries.csv')
    df['gasDayStart'] = pd.to_datetime(df['gasDayStart'])
    agsi_metrics = [
        'gasInStorage',
        'full',
        'injection',
        'withdrawal',
        'workingGasVolume'
    ]

    df['net_flow'] = -1 * df['netWithdrawal']
    df['days_of_supply'] = df['gasInStorage'] / df['consumption']

    df['month'] = df['gasDayStart'].dt.month
    df['year'] = df['gasDayStart'].dt.year
    df['season'] = pd.cut(
        df['gasDayStart'].dt.month, 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )

    df['post_invasion'] = df['gasDayStart'] >= '2022-02-24'
    df['post_sanctions'] = df['gasDayStart'] >= '2022-06-03'  # EU 6th sanctions package

    stats_by_country = df.groupby(['name', 'post_invasion'])[agsi_metrics + ['net_flow']].agg(['mean', 'std', 'min', 'max'])
    print(stats_by_country)

    # 2. Storage level trends
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, 
        x='gasDayStart', 
        y='full', 
        hue='name',
        marker='o',
        markersize=4,
        markevery=30  # Plot marker every 30 days to avoid overcrowding
    )
    plt.axvline(x=pd.to_datetime('2022-02-24'), color='red', linestyle='--', label='Invasion')
    plt.axvline(x=pd.to_datetime('2022-06-03'), color='orange', linestyle='--', label='EU Sanctions')
    plt.title('Gas Storage Levels by Country')
    plt.ylabel('Storage Capacity Filled (%)')
    plt.legend()
    plt.savefig('storage_levels_by_country.png')
    plt.show()

    # 3. Injection/Withdrawal patterns
    monthly_flows = df.groupby(['name', 'year', 'month'])[['injection', 'withdrawal', 'net_flow']].mean().reset_index()

    plt.figure(figsize=(14, 8))
    for i, country in enumerate(df['name'].unique()):
        plt.subplot(2, 2, i+1)
        country_data = monthly_flows[monthly_flows['name'] == country]
        
        # Create a date for plotting (15th of each month)
        country_data['date'] = pd.to_datetime(country_data['year'].astype(str) + '-' + 
                                            country_data['month'].astype(str) + '-15')
        
        plt.plot(country_data['date'], country_data['net_flow'], 'b-o')
        plt.axvline(x=pd.to_datetime('2022-02-15'), color='red', linestyle='--')
        plt.title(f'{country} - Net Gas Flow')
        plt.ylabel('Net Flow (injection - withdrawal)')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig('gas_flows_by_country.png')
    plt.show()

    # 4. Seasonal patterns and year-over-year comparison
    seasonal_comparison = df.pivot_table(
        index=['name', 'month'],
        columns=['year'],
        values='full',
        aggfunc='mean'
    ).reset_index()

    plt.figure(figsize=(12, 8))
    for i, country in enumerate(df['name'].unique()):
        plt.subplot(2, 2, i+1)
        country_seasonal = seasonal_comparison[seasonal_comparison['name'] == country]
        
        # Plot each year as a separate line
        for year in [2020, 2021, 2022]:
            if year in country_seasonal.columns:
                plt.plot(country_seasonal['month'], country_seasonal[year], 
                        label=str(year), marker='o')
        
        plt.title(f'{country} - Seasonal Storage Pattern')
        plt.xlabel('Month')
        plt.ylabel('Storage Level (%)')
        plt.ylim(0, 100)
        plt.legend()
    plt.tight_layout()
    plt.savefig('seasonal_patterns.png')
    plt.show()

    # First, verify your post_invasion flag calculation
    print("Earliest date in dataset:", df['gasDayStart'].min())
    print("Latest date in dataset:", df['gasDayStart'].max())
    print("Number of rows before invasion date:", len(df[df['gasDayStart'] < '2022-02-24']))
    print("Number of rows on or after invasion date:", len(df[df['gasDayStart'] >= '2022-02-24']))

    # If your data only starts from 2022 or near the invasion date, you need earlier data
    # If you can't get earlier data, consider a different analysis approach

    # Fix the post_invasion flag
    invasion_date = pd.to_datetime('2022-02-24')
    df['post_invasion'] = df['gasDayStart'] >= invasion_date

    # Recalculate pre/post invasion datasets
    pre_invasion = df[df['gasDayStart'] < invasion_date]
    post_invasion = df[df['gasDayStart'] >= invasion_date]

    # Check again
    print("Pre-invasion data shape (after fix):", pre_invasion.shape)
    print("Post-invasion data shape (after fix):", post_invasion.shape)

    # If you still don't have pre-invasion data, you could try:
    # 1. Compare early-war vs late-war periods instead
    early_war = df[(df['gasDayStart'] >= invasion_date) & 
                (df['gasDayStart'] < invasion_date + pd.Timedelta(days=90))]
    late_war = df[df['gasDayStart'] >= invasion_date + pd.Timedelta(days=90)]

    # 2. Or compare by seasons or quarters in 2022
    df['quarter'] = df['gasDayStart'].dt.quarter
    q1_2022 = df[(df['year'] == 2022) & (df['quarter'] == 1)]
    q2_2022 = df[(df['year'] == 2022) & (df['quarter'] == 2)]
    q3_2022 = df[(df['year'] == 2022) & (df['quarter'] == 3)]
    q4_2022 = df[(df['year'] == 2022) & (df['quarter'] == 4)]

    # 3. Handle missing oil price data before correlation analysis
    df['brent_price_usd'] = df['brent_price_usd'].ffill()  # Forward fill missing values

def correlation_analysis():
    countries = ['DE', 'FR', 'PL', 'UA']
    for country in countries:
        df = pd.read_csv(f'data/agsi_data_{country}_complete.csv')
        df['gasDayStart'] = pd.to_datetime(df['gasDayStart'])
        df['gasInStorage'] = pd.to_numeric(df['gasInStorage'], errors='coerce')
        df['consumption'] = pd.to_numeric(df['consumption'], errors='coerce')
        df['injection'] = pd.to_numeric(df['injection'], errors='coerce')
        df['withdrawal'] = pd.to_numeric(df['withdrawal'], errors='coerce')
        df['workingGasVolume'] = pd.to_numeric(df['workingGasVolume'], errors='coerce')

        df = df[df['gasDayStart'] >= '2020-01-01']
        df = df.dropna(subset=['gasInStorage', 'consumption', 'injection', 'withdrawal', 'workingGasVolume'])

        # Calculate correlations
        corr = df[['gasInStorage', 'consumption', 'injection', 'withdrawal', 'workingGasVolume']].corr()
        print(f"\n--- {country} Correlation Matrix ---")
        print(corr)

        # Visualize correlations
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'{country} - Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'correlation_{country}.png')
        plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load your complete datasets
countries = ['DE', 'FR', 'PL', 'UA']
dfs = {}

for country in countries:
    try:
        file_path = f'data/agsi_data_{country}_complete.csv'
        dfs[country] = pd.read_csv(file_path)
        dfs[country]['gasDayStart'] = pd.to_datetime(dfs[country]['gasDayStart'])
        print(f"Loaded {country} data: {len(dfs[country])} rows from {dfs[country]['gasDayStart'].min()} to {dfs[country]['gasDayStart'].max()}")
    except Exception as e:
        print(f"Error loading {country} data: {e}")

# Load oil price data
try:
    oil_df = pd.read_excel('data/eia_oil.xls', sheet_name='Data 1', skiprows=2)
    oil_df['Date'] = pd.to_datetime(oil_df['Date'])
    oil_df['brent_price_usd'] = oil_df['Europe Brent Spot Price FOB (Dollars per Barrel)']
    print(f"Loaded oil data: {len(oil_df)} rows from {oil_df['Date'].min()} to {oil_df['Date'].max()}")
except Exception as e:
    print(f"Error loading oil data: {e}")

# Merge oil data with each country's AGSI data
merged_dfs = {}
for country, df in dfs.items():
    merged = pd.merge(
        df,
        oil_df,
        how='left',
        left_on='gasDayStart',
        right_on='Date'
    )
    
    # Forward fill missing oil prices (for weekends/holidays)
    merged['brent_price_usd'] = merged['brent_price_usd'].ffill()
    
    merged_dfs[country] = merged
    print(f"Merged {country} data: {len(merged)} rows")

# Define the invasion date
invasion_date = pd.to_datetime('2022-02-24')

# Correlation analysis - pre vs. post invasion
correlation_results = []

for country, df in merged_dfs.items():
    # Replace '-' and other non-numeric values with NaN
    for col in ['full', 'brent_price_usd', 'gasInStorage', 'injection', 'withdrawal']:
        if col in df.columns:
            # Replace non-numeric values with NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in critical columns for correlation
    clean_df = df.dropna(subset=['full', 'brent_price_usd'])
    merged_dfs[country] = clean_df
    print(f"Cleaned {country} data: {len(clean_df)} rows (removed {len(df) - len(clean_df)} rows with invalid values)")

# Now repeat your correlation analysis with the cleaned data
correlation_results = []
invasion_date = pd.to_datetime('2022-02-24')

for country, df in merged_dfs.items():
    # Split into pre and post invasion
    pre_invasion = df[df['gasDayStart'] < invasion_date]
    post_invasion = df[df['gasDayStart'] >= invasion_date]
    
    result = {'country': country}
    
    # Calculate correlations if enough data
    if len(pre_invasion) >= 30:
        pre_corr = pre_invasion['full'].corr(pre_invasion['brent_price_usd'])
        result['pre_invasion_corr'] = pre_corr
        print(f"{country} pre-invasion correlation: {pre_corr:.4f} (n={len(pre_invasion)})")
    else:
        result['pre_invasion_corr'] = np.nan
        print(f"{country} has insufficient pre-invasion data: {len(pre_invasion)} rows")
    
    if len(post_invasion) >= 30:
        post_corr = post_invasion['full'].corr(post_invasion['brent_price_usd'])
        result['post_invasion_corr'] = post_corr
        print(f"{country} post-invasion correlation: {post_corr:.4f} (n={len(post_invasion)})")
    else:
        result['post_invasion_corr'] = np.nan
        print(f"{country} has insufficient post-invasion data: {len(post_invasion)} rows")
    
    # Calculate change if both periods have data
    if not np.isnan(result.get('pre_invasion_corr', np.nan)) and not np.isnan(result.get('post_invasion_corr', np.nan)):
        result['correlation_change'] = result['post_invasion_corr'] - result['pre_invasion_corr']
        result['has_complete_data'] = True
    else:
        result['has_complete_data'] = False
    
    correlation_results.append(result)

# Convert to DataFrame and display
corr_df = pd.DataFrame(correlation_results)
print("\nCorrelation Analysis Results:")
print(corr_df)

# Visualize correlations
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Plot pre and post invasion correlations
countries_with_data = corr_df[corr_df['has_complete_data']].copy()

if not countries_with_data.empty:
    plt.figure(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(len(countries_with_data))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, countries_with_data['pre_invasion_corr'], width, label='Pre-Invasion')
    plt.bar(x + width/2, countries_with_data['post_invasion_corr'], width, label='Post-Invasion')
    
    # Add labels and formatting
    plt.xlabel('Country')
    plt.ylabel('Correlation Coefficient')
    plt.title('Oil Price vs Gas Storage Correlation: Before and After Invasion')
    plt.xticks(x, countries_with_data['country'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add correlation values as text
    for i, row in enumerate(countries_with_data.itertuples()):
        plt.text(i - width/2, row.pre_invasion_corr + 0.05, f"{row.pre_invasion_corr:.2f}", 
                ha='center', va='bottom', rotation=0)
        plt.text(i + width/2, row.post_invasion_corr + 0.05, f"{row.post_invasion_corr:.2f}", 
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('correlation_comparison.png', dpi=300)
    plt.show()
else:
    print("No countries have complete data for visualization")

# For Ukraine or other countries without pre-invasion data
# Do early-war vs late-war analysis
for country, df in merged_dfs.items():
    if country in ['UA'] or corr_df.loc[corr_df['country'] == country, 'has_complete_data'].iloc[0] == False:
        # Split the post-invasion period
        post_invasion = df[df['gasDayStart'] >= invasion_date]
        
        if len(post_invasion) >= 60:  # Need sufficient data to split
            mid_date = invasion_date + pd.Timedelta(days=90)  # 3 months after invasion
            early_war = post_invasion[post_invasion['gasDayStart'] < mid_date]
            late_war = post_invasion[post_invasion['gasDayStart'] >= mid_date]
            
            if len(early_war) >= 30 and len(late_war) >= 30:
                early_corr = early_war['full'].corr(early_war['brent_price_usd'])
                late_corr = late_war['full'].corr(late_war['brent_price_usd'])
                
                print(f"\n{country} Alternative Analysis (Early War vs Late War):")
                print(f"Early War Correlation (First 3 months): {early_corr:.4f} (n={len(early_war)})")
                print(f"Late War Correlation (After 3 months): {late_corr:.4f} (n={len(late_war)})")
                print(f"Correlation Change: {late_corr - early_corr:.4f}")
