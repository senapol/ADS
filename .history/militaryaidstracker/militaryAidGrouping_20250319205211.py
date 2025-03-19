import pandas as pd
import numpy as np

file_path = "data/UkraineTracker.xlsx"
xls = pd.ExcelFile(file_path)
aid_df = pd.read_excel(xls, sheet_name="Bilateral Assistance, MAIN DATA")

# Got the exchange rates by getting the unique currencies
exchange_rates = {
    "AUD": 0.64,   # 1 AUD ~ 0.64 EUR
    "USD": 0.93,   # 1 USD ~ 0.93 EUR
    "EUR": 1.00,   # Reference currency
    "BGN": 0.51,   # 1 BGN ~ 0.51 EUR
    "CAD": 0.70,   # 1 CAD ~ 0.70 EUR
    "CNY": 0.13,   # 1 CNY ~ 0.13 EUR
    "HRK": 0.13,   # 1 HRK ~ 0.13 EUR
    "CZK": 0.042,  # 1 CZK ~ 0.042 EUR
    "DKK": 0.13,   # 1 DKK ~ 0.13 EUR
    "HUF": 0.0026, # 1 HUF ~ 0.0026 EUR
    "ISK": 0.0068, # 1 ISK ~ 0.0068 EUR
    "JPY": 0.0075, # 1 JPY ~ 0.0075 EUR
    "GBP": 1.16,   # 1 GBP ~ 1.16 EUR
    "NZD": 0.59,   # 1 NZD ~ 0.59 EUR
    "NOK": 0.088,  # 1 NOK ~ 0.088 EUR
    "PLN": 0.21,   # 1 PLN ~ 0.21 EUR
    "KRW": 0.00077,# 1 KRW ~ 0.00077 EUR
    "RON": 0.20,   # 1 RON ~ 0.20 EUR
    "SEK": 0.088,  # 1 SEK ~ 0.088 EUR
    "CHF": 1.02    # 1 CHF ~ 1.02 EUR
}

df = aid_df[['activity_id', 'announcement_date', 'donor', 'item_type', 'item_value_estimate_USD', 'explanation', 
                    'reporting_currency', 'source_reported_value', 'measure']].copy() # , 'classified_category'

# print(df['item_type'].unique())

dict_weapons = {'.' : None, 'Ammunition for portable defence system' : 'Portable defence system', 'Ammunition for heavy weapon' : 'Heavy weapon',
 'Ammunition for light infantry' : 'Light armaments & infantry', 
 'humanitarian' : 'Humanitarian', 'Ammunition for Heavy Weapon' : 'Heavy weapon', 'military equipment' : 'Military equipment',
 'Military equipment ' : 'Military equipment', 'Military Equipment' : 'Military equipment', 'Aviation and Drones' : 'Aviation and drones',
 'Heavy Weapon' : 'Heavy weapon', 'ammunition for heavy weapon' : 'Heavy weapon'} # 'Funding, training, services' 'Missile', 'Ammunition'

for key, value in dict_weapons.items():
    df.loc[df['item_type'] == key, 'item_type'] = value

print(df['item_type'].unique())

df['item_value_estimate_USD'] = df['item_value_estimate_USD'].replace({'.': np.nan, 'No price': np.nan}, regex=False)

print(df.loc[df['item_type'] == 'Ammunition', df['item_type'] == 'Funding, training, services', df['item_type'] == 'Missile'])
# -------
def convert_to_eur(amount, currency):
    """Helper that multiplies amount by the relevant exchange rate."""
    if pd.isna(amount) or amount == 'Not given' or amount == 'Not Given' or pd.isna(currency):
        return np.nan
    rate = exchange_rates.get(currency)
    # print(rate, amount)
    return amount * rate

# Convert the source reported value to EUR from the relevant reporting_currency, if not given return null
df.loc[:, "source_reported_value_EUR"] = df.apply(
    lambda row: convert_to_eur(row["source_reported_value"], row["reporting_currency"]),
    axis=1
)
