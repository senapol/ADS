import pandas as pd
import plotly.express as px
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default='constant', help='Mode for processing data: constant or percapita')
args = parser.parse_args()
print(args.mode)
if args.mode not in ['constant', 'percapita']:
    print(f'Invalid mode {args.mode}: Please enter either -m constant or -m percapita')
    exit()

# Variables based on data mode
if args.mode == 'constant':
    sheet_name = 'Constant (2022) US$'
    skiprows = 5
    ylabel = 'Military Expenditure in Millions (Constant 2022 US$) '
elif args.mode == 'percapita':
    sheet_name = 'Per capita'
    skiprows = 6
    ylabel = 'Military Expenditure per capita in Millions (Current US$)'


countries_of_interest = {
    'direct': ['Ukraine', 'Russia'],
    'neighbouring': ['Belarus', 'Moldova', 'Poland', 'Romania', 'Hungary', 'Slovakia'],
    'baltic': ['Estonia', 'Latvia', 'Lithuania'],
    'other': ['Germany', 'France', 'United Kingdom', 'United States of America', 'Finland']
}


def filter_relevant_countries(df, countries):
    'Filter df to only include countries of interest'

    df = df[df['Country'].isin(countries)]
    return df


def process_milex_excel(path):
    'Read and convert Milex xlsx file to pandas df'

    try:
        print(f'Reading file: {path}')
        df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows, na_values=['...', ''])

        print(df)
        return df

    except Exception as e:
        print(f'Error {e}: Could not read file')
        return None


def clean_df(df):
    
        # Systematically clean df
        df = df.drop(columns=[col for col in df.columns if ('Unnamed' in str(col) or 'Notes' in str(col))])
        df = df.dropna(subset=['Country'])

        # Filter to only include countries of interest

        df = filter_relevant_countries(df, countries_of_interest['direct'] + countries_of_interest['other'])

        return df


def plot_line(df):
     
    # Only include data from years from 2010 onwards
    year_columns = [col for col in df.columns if str(col).isnumeric() and pd.to_numeric(col) >= 2010]

    df = df.melt(id_vars = ['Country'], value_vars=year_columns, var_name='Year', value_name='Milex')

    df = df[df['Country'] != 'United States of America']

    fig = px.line(df, x='Year', y='Milex', title='Military Expenditure over Time', color='Country', labels={'Milex': ylabel, 'Year': 'Year'})
    fig.show()


df = process_milex_excel('data/SIPRI-Milex-data-1948-2023.xlsx')
df = clean_df(df)
plot_line(df)
