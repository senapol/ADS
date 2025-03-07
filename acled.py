# https://acleddata.com/knowledge-base/codebook/#acled-data-columns-at-a-glance

import pandas as pd

df = pd.read_csv('data/Ukraine_Black_Sea_2020_2025_Feb28.csv')
pd.set_option('display.max_columns', None)

# print(df.columns[df.isna().any()])
print(df.head(5))