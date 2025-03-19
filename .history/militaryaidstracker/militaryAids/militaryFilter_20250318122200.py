import pandas as pd

# Load the dataset
file_path = "data/UkraineTracker.xlsx"
xls = pd.ExcelFile(file_path)

# Load the main data sheet
main_data_sheet = "Bilateral Assistance, MAIN DATA"
df = pd.read_excel(xls, sheet_name=main_data_sheet)

# Define refined categories for classification
category_keywords = {
    "Air Defense": ["missile", "air defense", "anti-aircraft", "radar", "patriot"],
    "Ground Combat": ["tank", "armored vehicle", "infantry", "rifle", "combat"],
    "Artillery & Heavy Weapons": ["howitzer", "rocket launcher", "artillery", "mortar", "cannon", "heavy", "carrier", "armored", "vehicle", "arm", "weapon package"],
    "Ammunition & Explosives": ["ammunition", "grenade", "explosives", "munition"],
    "Financial & Economic Aid": ["loan", "billion", "million", "eur", "usd", "economic", "budget", "fund allocation", "mfa", "financial package", "economic support"],
    "Training & Logistics & Support": ["training", "instructor", "advisors", "exercise", "education", "transport", "medical", "fuel", "supplies", "logistics"],
    "Fighter Jets & Airforce": ["fighter jet", "mig", "f-16", "aircraft", "helicopter", "jet", "bomber"],
    "Military Commitments": ["military aid commitments", "defense support", "military support"],
    "General Announcements": ["announced", "pledged", "donated", "allocated"]
}

# Function to classify each row based on keywords in the explanation column
def classify_explanation(text):
    if pd.isna(text):
        return "Uncategorised"

    text_lower = text.lower()
    for category, keywords in category_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return category
    return "Uncategorised"

# Apply classification function to the dataset
df["classified_category"] = df["explanation"].apply(classify_explanation)

# **Save full categorized dataset**
output_file_path = "data/UkraineTracker_Categorized.xlsx"
with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Categorized Data", index=False)

print(f"Updated file saved as: {output_file_path}")

# **Extract only non-Uncategorised rows**
df_filtered = df[df["classified_category"] != "Uncategorised"]

# Save filtered data (only classified categories)
filtered_output_path = "data/UkraineTracker_Categorized_OnlyClassified.xlsx"
with pd.ExcelWriter(filtered_output_path, engine="xlsxwriter") as writer:
    df_filtered.to_excel(writer, sheet_name="Filtered Data", index=False)

print(f"Filtered data saved as: {filtered_output_path}")

# Optionally, save a CSV file for quick preview
df_filtered[["explanation", "classified_category"]].dropna().to_csv("classified_aid_categories_filtered.csv", index=False)
print("Filtered sample data saved as classified_aid_categories_filtered.csv")
