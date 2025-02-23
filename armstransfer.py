import pandas as pd
import numpy as np
import plotly.express as px

def process_at_csv(path):
    'Read and convert Arms Transfer xlsx file to pandas df'

    try:
        print(f'Reading file: {path}')
        df = pd.read_csv(path, skiprows=11, na_values=['...', ''])

        print(df)
        return df

    except Exception as e:
        print(f'Error {e}: Could not read file')
        return None

df = process_at_csv('data/trade-register.csv')

df = df.groupby('Supplier')['Numbers delivered'].sum().reset_index()
df_sorted = df.sort_values(by='Numbers delivered', ascending=True).tail(15)

fig = px.bar(df_sorted, x='Supplier', y='Numbers delivered', title='Top 10 Arms Suppliers to Ukraine (2000-2024)', labels={'Arms imports': 'Arms Imports', 'Country': 'Country'})
fig.show()