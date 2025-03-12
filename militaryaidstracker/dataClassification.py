# Note
# divided sectors into: (1) military (2) financial (3) humanitarian (4) presidential drawdowns
# did keyword extraction for data classification

import pandas as pd

file_path = "data/UkraineTracker.xlsx"
xls = pd.ExcelFile(file_path)

main_data_sheet = "Bilateral Assistance, MAIN DATA"
df = pd.read_excel(xls, sheet_name=main_data_sheet)

# Define classification keywords for each category
category_keywords = {
    "Air Defense": ["missile", "air defense", "anti-aircraft", "radar", "patriot"],
    "Ground Combat": ["tank", "armored vehicle", "infantry", "rifle", "combat"],
    "Artillery & Heavy Weapons": ["howitzer", "rocket launcher", "artillery", "mortar", "cannon"],
    "Ammunition & Explosives": ["ammunition", "grenade", "explosives", "munition"],
    "Logistics & Support": ["transport", "medical", "fuel", "supplies", "logistics"],
    "Humanitarian Aid": ["humanitarian", "food", "medical aid", "assistance", "relief"],
    "Financial & Economic Aid": ["loan", "billion", "million", "eur", "usd", "economic", "budget", "fund allocation", "mfa", "financial package", "economic support"],
    "Training & Advisory": ["training", "instructor", "advisors", "exercise", "education"],
    "Fighter Jets & Airforce": ["fighter jet", "mig", "f-16", "aircraft", "helicopter", "jet", "bomber"],
    "Military Commitments": ["military aid commitments", "defense support", "military support", "weapon package"],
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

# Save the updated dataframe back to an Excel file
output_file_path = "data/UkraineTracker_Categorized.xlsx"
with pd.ExcelWriter(output_file_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="Categorized Data", index=False)

print(f"Updated file saved as: {output_file_path}")

# Optionally, save a CSV file for quick viewing
df[["explanation", "classified_category"]].dropna().to_csv("classified_aid_categories.csv", index=False)
print("Sample data saved as classified_aid_categories.csv")
