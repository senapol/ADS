import pandas as pd
import numpy as np
# import ace_tools as tools

# frontline_events_path = "data/filtered_frontline_events.csv"
# frontline_df = pd.read_csv(frontline_events_path)

aid_data_path = "data/UkraineTracker_Categorized_OnlyClassified.xlsx"
aid_df = pd.read_excel(aid_data_path, sheet_name=None)  # Load all sheets

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

# frontline_df['date'] = pd.to_datetime(frontline_df['date'], errors='coerce')
# frontline_df = frontline_df.drop_duplicates().reset_index(drop=True)
# frontline_cleaned = frontline_df[['date', 'latitude', 'longitude', 'event_type', 'location', 'admin1', 'description']]
aid_main_df = aid_df['Filtered Data'].head(300)
# aid_main_df['announcement_date'] = pd.to_datetime(aid_main_df['announcement_date'], errors='coerce')
# aid_main_df['source_reported_value'] = pd.to_numeric(aid_main_df['source_reported_value'], errors='coerce')

df = aid_main_df[['announcement_date', 'donor', 'aid_type_general', 'aid_type_specific',
                           'explanation', 'reporting_currency', 'source_reported_value', 'classified_category', 'measure']]

print(df['reporting_currency'].unique())

def convert_to_eur(amount, currency):
    """Helper that multiplies amount by the relevant exchange rate."""
    if pd.isna(amount) or pd.isna(currency):
        return np.nan
    rate = exchange_rates.get(currency)
    print(rate, currency)
    return amount * rate

df["source_reported_value_EUR"] = df.apply(
    lambda row: convert_to_eur(row["source_reported_value"], row["reporting_currency"]),
    axis=1
)

df.head(50)
# aid_cleaned = aid_cleaned.dropna(subset=['announcement_date', 'aid_type_specific', 'source_reported_value'])
# aid_cleaned['date'] = pd.to_datetime(aid_cleaned['announcement_date'])  # Ensure 'date' column is in datetime format
# aid_cleaned = aid_cleaned.sort_values(by='date')  # Sort by date in ascending order
# aid_cleaned = aid_cleaned.reset_index(drop=True)  # Reset index after sorting

# # frontline_cleaned = frontline_cleaned.dropna(subset=['date', 'event_type'])

# # frontline_cleaned = frontline_cleaned.reset_index(drop=True)
# aid_cleaned = aid_cleaned.reset_index(drop=True)

# # Save the cleaned datasets as new files

# # cleaned_frontline_path = "data/cleaned/cleaned_frontline_events.csv"
# cleaned_aid_path = "data/cleaned/cleaned_military_aid.csv"

# # Save cleaned datasets as CSV files
# # frontline_cleaned.to_csv(cleaned_frontline_path, index=False)
# aid_cleaned.to_csv(cleaned_aid_path, index=False)

# cleaned_frontline_path, cleaned_aid_path


# import pandas as pd
# import numpy as np

# # -----------------------------------------------------------------------------
# # 1. Toy Exchange Rate Lookup
# #    In reality, you'd maintain a more complete mapping or dynamically fetch rates.
# # -----------------------------------------------------------------------------
# exchange_rates = {
#     "EUR": 1.0,
#     "USD": 0.93,  # Example: 1 USD ~ 0.93 EUR
#     "GBP": 1.16, # Example: 1 GBP ~ 1.16 EUR
#     # ... add other currencies as needed
# }

# # -----------------------------------------------------------------------------
# # 2. Sample DataFrame (as an example)
# #    Suppose you have columns like:
# #      - activity_id
# #      - source_reported_value
# #      - reporting_currency
# #      - item_value_estimate_USD
# # -----------------------------------------------------------------------------
# data = {
#     "activity_id": [1, 1, 2, 2, 3],
#     "source_reported_value": [100_000, None, None, None, 500_000],
#     "reporting_currency": ["USD", None, None, None, "EUR"],
#     "item_value_estimate_USD": [None, 5_000, 10_000, 15_000, None],
# }
# df = pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 3. Convert source_reported_value to EUR (where present)
#    We'll create a new column 'source_reported_value_EUR'
# -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
# # 4. Group by activity_id & create final tot_activity_value_EUR
# # -----------------------------------------------------------------------------
# def aggregate_tot_value_eur(group):
#     """
#     - If there's a non-null source_reported_value_EUR in the group, use that (assuming
#       it applies to the entire activity).
#     - Otherwise, sum item_value_estimate_USD across the group and convert that sum to EUR.
#     """
#     # Check any non-null source_reported_value_EUR
#     non_null_vals = group["source_reported_value_EUR"].dropna()
#     if not non_null_vals.empty:
#         # Use the first non-null (or you could take max if you prefer)
#         tot_eur = non_null_vals.iloc[0]
#     else:
#         # Sum all item_value_estimate_USD
#         total_usd = group["item_value_estimate_USD"].sum(min_count=1)
#         # Convert that sum to EUR
#         tot_eur = total_usd * exchange_rates["USD"] if not np.isnan(total_usd) else np.nan

#     # Return a 1-row Series with your final total
#     return pd.Series({
#         "tot_activity_value_EUR": tot_eur
#     })

# # Apply the function to each group
# aggregated_df = df.groupby("activity_id").apply(aggregate_tot_value_eur).reset_index()

# # -----------------------------------------------------------------------------
# # 5. (Optional) Merge back or keep as your final table
# # -----------------------------------------------------------------------------
# # If you need just one final DataFrame with activity_id and tot_activity_value_EUR:
# final_df = aggregated_df.copy()

# # Display results
# print("Original DF:")
# print(df, "\n")
# print("Aggregated DF (one row per activity_id):")
# print(final_df)
