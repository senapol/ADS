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

df = aid_df[['activity_id', 'sub_activity_id', 'aid_type_general', 'total_value_dummy', 'announcement_date', 'donor', 'item_type', 'item_value_estimate_USD', 'explanation', 
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

df['item_value_estimate_USD'] = pd.to_numeric(df['item_value_estimate_USD'], errors='coerce')

# explanations = df.loc[~((df['aid_type_general'] == "Humanitarian") | (df['aid_type_general'] == "Financial") | ~(df['item_type'].isna()))]['explanation']

# # Save the cleaned datasets as new files
# cleaned_aid_path = "data/cleaned/explanations_to_filter.csv"

# explanations.to_csv(cleaned_aid_path, index=False)

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

# df.drop(columns={'source_reported_value'}, inplace=True)

# print(df.loc[((df['aid_type_general'].astype(str) == 'Financial') | (df['aid_type_general'].astype(str) == 'Humanitarian')) & (df['item_type'] != None)])

# --------
# Create a collumn for each aid category and sum them for each sub group(that has same sub_activity_id)
# and also sum the the reported source value
for aid_type in aid_categories:
    df[aid_type] = np.where(
        (df["item_type"] == aid_type) & df['item_value_estimate_USD'],
        df['item_value_estimate_USD'] * exchange_rates['USD'],
        0
    )

def aggregate_tot_value_eur(group):

    """ 
    This sums all elements in the aid categories, so we have the totalamount for each category 
    """

    # Sum all row values for each aid category
    for aid_type in aid_categories:
        total_eur = float(group[aid_type].sum(min_count=1)) if not np.isnan(float(group[aid_type].sum(min_count=1))) else np.nan
        group.loc[:, aid_type] = total_eur

    # The total_value dummy prevent double counting
    # group['source_reported_value_EUR'] *= group['total_value_dummy']
    # group['source_reported_value_EUR'] = group['source_reported_value_EUR'].sum()

    # Sum all item_value_estimate_USD
    total_usd = float(group["item_value_estimate_USD"].sum(min_count=1))
    # Convert that sum to EUR
    tot_eur = float(total_usd * exchange_rates["USD"]) if not np.isnan(total_usd) else 0

    group.loc[:, "items_value_estimate_EUR"] = tot_eur

    row_out = group.head(1).copy()
    s_reported = group["source_reported_value_EUR"].iloc[0]
    if pd.isna(s_reported):
        adjusted_value = 0
    else:
        adjusted_value = s_reported - tot_eur if (s_reported > tot_eur) and (tot_eur >= 0) else 0
    
    group.loc[:, "Uncategorised"] = adjusted_value

    # The reported_source_value is the same for data points with same sub_activity_id so it's okay to get the first value
    return group.head(1)

# Sum all the item_value_estimate_USD by the sub_activity_id 
# and return just one row for every group since they have the same values for the columns we need
df = df.groupby("sub_activity_id").apply(aggregate_tot_value_eur)

# print(df.head(15))

def aggregate_tot_value_eur_2(group):

    # If flip all the total flip the children so just the allocations count and not the commitment
    if (group['measure'].iloc[0] == 'Commitment'):
        group.loc[group['measure'] == 'Commitment', 'total_value_dummy'] = 0 # 1 - group.loc[group['measure'] == 'Commitment', 'total_value_dummy']
        group.loc[group['measure'] != 'Commitment', 'total_value_dummy'] = 1 # 1 - group.loc[group['measure'] == 'Commitment', 'total_value_dummy']
        # group['total_value_dummy'] = 1 - group['total_value_dummy']
    
    # Sum all the clidren since they are all sub_activity values
    group['source_reported_value_EUR'] *= group['total_value_dummy']
    # group['source_reported_value_EUR'] = group['source_reported_value_EUR'].sum()
    group.loc[:, "Uncategorised"] = np.where(
        (group["source_reported_value_EUR"].notnull()) & 
        (group["source_reported_value_EUR"] > group["items_value_estimate_EUR"]),
        group["source_reported_value_EUR"] - group["items_value_estimate_EUR"],
        0
    )

    return group

df = df.groupby("activity_id").apply(aggregate_tot_value_eur_2)

# # If it's Financial aid, store the whole reported value in "Financial"
mask = df['aid_type_general'] == "Financial"

# Then subtract that out from "Uncategorised"
df.loc[mask, "Uncategorised"] = (
    df.loc[mask, "Uncategorised"] 
    - df.loc[mask, "source_reported_value_EUR"] + df.loc[mask, "Financial"] + 1
)

df.loc[mask, "Financial"] = df.loc[mask, "source_reported_value_EUR"]


# # If it is financial aid add the whole reported value subtracted by the items mentioned to prevent double counting
# df["Financial"] = np.where(
#     (df['aid_type_general'] == "Financial"),
#     df["source_reported_value_EUR"],
#     df['Financial']
# )

mask = df['aid_type_general'] == "Humanitarian"

df.loc[mask, "Uncategorised"] = (
    df.loc[mask, "Uncategorised"] - df.loc[mask, "source_reported_value_EUR"] + df.loc[mask, "Humanitarian"] + 1
)

df.loc[mask, "Humanitarian"] = df.loc[mask, "source_reported_value_EUR"]


# If it is humanitarian aid add the whole reported value subtracted by the items mentioned to prevent double counting
# df["Humanitarian"] = np.where(
#     (df['aid_type_general'] == "Humanitarian"),
#     df["source_reported_value_EUR"],
#     df['Humanitarian']
# )

# print(df.head(15))

# Identify rows with no reported and item values
invalid_mask = df['source_reported_value_EUR'].isna() & df['items_value_estimate_EUR'].isna()

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
cols_to_sum = aid_categories
# print(df[cols_to_sum + ['announcement_date']].head(30))

# grouped = df.groupby(pd.Grouper(key='announcement_date', freq='W'))[cols_to_sum + ['Uncategorised'] + ["source_reported_value_EUR"]]

#   # adjust this to the exact group key as it appears in the groups
# target_date = pd.Timestamp('2024-09-01')
# # Retrieve the group for that date
# group = grouped.get_group(target_date)
# print(group)

df = df.groupby([pd.Grouper(key='announcement_date', freq='W')])[cols_to_sum + ['Uncategorised']].sum().reset_index() # .sum().reset_index()

df = df.loc[(df["announcement_date"] == pd.Timestamp('2024-09-01'), "Uncategorised")] = 0

# 1. Define your list of columns to sum
# print(df[cols_to_sum].dtypes)

# 2. Sum them row-wise and store in a new column, e.g. "total_value_EUR"
print(df[cols_to_sum + ['Uncategorised']].sum()/1000000000)

# print(df.loc[df.loc[df['aid_type_general'].astype(str) == "Humanitarian", 'source_reported_value_EUR'].max() == df['source_reported_value_EUR']])
# print(df.head(30))

# print(df.count())
# print(df.head(20))

# Save the cleaned datasets as new files
cleaned_aid_path = "data/cleaned/aid_categories_weekly.csv"

df.to_csv(cleaned_aid_path, index=False)