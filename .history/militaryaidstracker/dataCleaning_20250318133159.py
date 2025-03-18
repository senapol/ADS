import pandas as pd
# import ace_tools as tools

# frontline_events_path = "data/filtered_frontline_events.csv"
# frontline_df = pd.read_csv(frontline_events_path)

aid_data_path = "data/UkraineTracker_Categorized_OnlyClassified.xlsx"
aid_df = pd.read_excel(aid_data_path, sheet_name=None)  # Load all sheets

# frontline_df['date'] = pd.to_datetime(frontline_df['date'], errors='coerce')
# frontline_df = frontline_df.drop_duplicates().reset_index(drop=True)
# frontline_cleaned = frontline_df[['date', 'latitude', 'longitude', 'event_type', 'location', 'admin1', 'description']]
aid_main_df = aid_df['Filtered Data']
# aid_main_df['announcement_date'] = pd.to_datetime(aid_main_df['announcement_date'], errors='coerce')
# aid_main_df['source_reported_value'] = pd.to_numeric(aid_main_df['source_reported_value'], errors='coerce')

aid_cleaned = aid_main_df[['announcement_date', 'donor', 'aid_type_general', 'aid_type_specific',
                           'explanation', 'reporting_currency', 'source_reported_value', 'classified_category', 'measure']]

print(aid_cleaned['reporting_currency'].unique())

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
