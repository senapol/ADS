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

df = aid_df[['activity_id', 'aid_type_general', 'total_value_dummy', 'announcement_date', 'donor', 'item_type', 'item_value_estimate_USD', 'explanation', 
                    'reporting_currency', 'source_reported_value', 'measure']].copy() # , 'classified_category'

aid_categories = [
    "Humanitarian",
    "Military equipment",
    "Aviation and drones",
    "Portable defence system",
    "Heavy weapon",
    "Financial"
]

# Note: Ammunition is set to none because there isn't a value estimate in any of the ammunition rows
dict_weapons = {'.' : None, 'Ammunition for portable defence system' : 'Portable defence system', 'Ammunition for heavy weapon' : 'Heavy weapon',
 'Ammunition for light infantry' : "Military equipment", 
 'humanitarian' : 'Humanitarian', 'Ammunition for Heavy Weapon' : 'Heavy weapon', 'military equipment' : 'Military equipment',
 'Military equipment ' : 'Military equipment', 'Military Equipment' : 'Military equipment', 'Aviation and Drones' : 'Aviation and drones',
 'Heavy Weapon' : 'Heavy weapon', 'ammunition for heavy weapon' : 'Heavy weapon', 'Missile' : 'Heavy weapon',
 'Funding, training, services' : 'Military equipment',  'Ammunition' : None, "Light armaments & infantry" : "Military equipment"} #

# Change the item type so it keeps only the aid_categories
for key, value in dict_weapons.items():
    df.loc[df['item_type'] == key, 'item_type'] = value

# The count is 4044 is the whole dataset. So they either have an item or it is humanitarian or financial
# print(df[~df['item_type'].isna() | (df['aid_type_general'] != "Financial") | (df["aid_type_general"] != "Humanitarian")].count())
# print(df.count())

# Make the missing values null
df['item_value_estimate_USD'] = df['item_value_estimate_USD'].replace({'.': np.nan, 'No price': np.nan}, regex=False)

# Make a column for every type and take the item_value estimate in EUROS
for aid_type in aid_categories:
    df[aid_type] = np.where(
        (df["item_type"] == aid_type) & df['item_value_estimate_USD'],
        df['item_value_estimate_USD'] * exchange_rates['USD'],
        0
    )

# Exclude commitment, which are just promises, since the allocations can be a subset of the commitment
# invalid_mask = (df['measure'] == "Commitment")
# df = df[~invalid_mask]

# -------
def convert_to_eur(amount, currency):
    """Helper that multiplies amount by the relevant exchange rate."""
    if pd.isna(amount) or amount == 'Not given' or amount == 'Not Given' or pd.isna(currency):
        return np.nan
    rate = exchange_rates.get(currency)
    return amount * rate

# Convert the source reported value to EUR from the relevant reporting_currency, if not given return null
df.loc[:, "source_reported_value_EUR"] = df.apply(
    lambda row: convert_to_eur(row["source_reported_value"], row["reporting_currency"]),
    axis=1
)

# --------
def aggregate_tot_value_eur(group):

    """ 
    This sums all elements in the aid categories, so we have the totalamount for each category 
    """

    for aid_type in aid_categories:
        # Sum all row values
        total_usd = float(group[aid_type].sum(min_count=1))
        # Convert that sum to EURO
        tot_eur = int(total_usd * exchange_rates["USD"]) if not np.isnan(total_usd) else np.nan
        group.loc[:, aid_type] = tot_eur

    """
    - If there's a non-null source_reported_value_EUR in the group, use that (assuming
      it applies to the entire activity).
    - Otherwise, sum item_value_estimate_USD across the group and convert that sum to EUR.
    """

    if (group['measure'].iloc[0] == 'Commitment'):
        group['total_value_dummy'] = ~group['total_value_dummy']
    
    group['source_reported_value_EUR'] *= group['total_value_dummy']
    group['source_reported_value_EUR'] = group['source_reported_value_EUR'].sum()

    # Check any non-null source_reported_value_EUR
    # else:
    # Sum all item_value_estimate_USD
    total_usd = float(group["item_value_estimate_USD"].sum(min_count=1))
    # Convert that sum to EUR
    tot_eur = int(total_usd * exchange_rates["USD"]) if not np.isnan(total_usd) else np.nan

    group.loc[:, "items_value_estimate_EUR"] = tot_eur

    non_null_vals = group[~group["source_reported_value_EUR"].isna()]
    if non_null_vals.empty:
            group.loc[:, "source_reported_value_EU"] = tot_eur

    return group.head(1)

# If there isn't a source_reported_value_EUR sum all the item_value_estimate_USD by the activity_id 
# and return just one row for every group since they have the same activity_id
df = df.groupby("activity_id").apply(aggregate_tot_value_eur)

# NOTE: This might need changing
df["Financial"] = np.where(
    df['aid_type_general'] == "Financial",
    df["source_reported_value_EUR"] - df['items_value_estimate_EUR'],
    df['Financial']
)

df["Humanitarian"] = np.where(
    df['aid_type_general'] == "Humanitarian",
    df["source_reported_value_EUR"] - df['items_value_estimate_EUR'],
    df['Humanitarian']
)

# Identify rows with no reported and item values
invalid_mask = df['source_reported_value_EUR'].isna() & df['item_value_estimate_USD'].isna()

# Count how many rows is that
both_null_count = df[invalid_mask].shape[0]
print("Rows with both columns null:", both_null_count)
df.drop(columns={"item_value_estimate_USD", "source_reported_value"}, inplace=True)

# Keep the rows that are not invalid
df = df[~invalid_mask]

# ---------
# First attempt: Convert the announcement_date to datetime, with null set when there is an error with the coerce parameter
df["announcement_date_converted"] = pd.to_datetime(df["announcement_date"], errors='coerce')
# Identify rows where the conversion failed (NaT in the converted column)
invalid_mask = df["announcement_date_converted"].isna()
# For those rows, clean the string (remove characters and whitespaces), then re-convert to datetime
df.loc[invalid_mask, "announcement_date"] = pd.to_datetime(
    df.loc[invalid_mask, "announcement_date"]
    .astype(str)
    .str.replace(r'[A-Za-z]', '', regex=True) # .str.replace("until ", "", regex=False)
    .str.replace(r'\s+', '', regex=True),
    errors='ignore'
)
# Drop the helping column
df.drop(columns="announcement_date_converted", inplace=True)

# I checked the values that were not converted to date and changed them manually
dict_invalid = {"ESM17" : "6/30/2023", "ESM7" : "6/30/2022", "FRM13" : "01/01/2023", "JPH10" : "1/1/2023", "LUH8" : "1/1/2024", "TRH3" : "3/20/2022"}
for (key, value) in dict_invalid.items():
    df.loc[df['activity_id'] == key, 'announcement_date'] = value

# 1 value droped due to unknown date
df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors='coerce')
df = df.dropna(subset=['announcement_date'])

df = df.sort_values(by='announcement_date')

# 1. Define your list of columns to sum
cols_to_sum = aid_categories + ['source_reported_value_EUR']

# print(df[cols_to_sum].dtypes)

# 2. Sum them row-wise and store in a new column, e.g. "total_value_EUR"
print(df[cols_to_sum].sum()/1000000000)

# print(df.count())
# print(df.head(20))

# # Save the cleaned datasets as new files
# cleaned_aid_path = "data/cleaned/military_aid_NEW.csv"

# df.to_csv(cleaned_aid_path, index=False)