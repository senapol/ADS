import os
import pandas as pd
import matplotlib.pyplot as plt

fig_dir = "Figures"
os.makedirs(fig_dir, exist_ok=True)




file_path = "data/UkraineTracker.xlsx"  # Update with your file path
xls = pd.ExcelFile(file_path)
df_main_data = pd.read_excel(xls, sheet_name="Bilateral Assistance, MAIN DATA")
df_main_data["announcement_date"] = pd.to_datetime(df_main_data["announcement_date"], errors='coerce')

#  Number of aids per country
aid_per_country = df_main_data["donor"].value_counts()

#  Number of aids per aid category
aid_per_category = df_main_data["aid_type_general"].value_counts()

# Number of aids per month
df_main_data["year_month"] = df_main_data["announcement_date"].dt.to_period("M")
aid_per_month = df_main_data["year_month"].value_counts().sort_index()

# Plot each graph
plt.figure(figsize=(10, 5))
aid_per_country.plot(kind="bar")
plt.title("Number of Aids per Country")
plt.xlabel("Country")
plt.ylabel("Number of Aids")
plt.xticks(rotation=90)
plt.savefig(os.path.join(fig_dir, "aids_per_country.png"))
plt.show()

plt.figure(figsize=(10, 5))
aid_per_category.plot(kind="bar")
plt.title("Number of Aids per Aid Category")
plt.xlabel("Aid Category")
plt.ylabel("Number of Aids")
plt.xticks(rotation=45)
plt.savefig(os.path.join(fig_dir, "aids_per_category.png"))
plt.show()

plt.figure(figsize=(12, 5))
aid_per_month.plot(kind="line", marker="o")
plt.title("Number of Aids per Month")
plt.xlabel("Month")
plt.ylabel("Number of Aids")
plt.xticks(rotation=45)
plt.grid()
plt.savefig(os.path.join(fig_dir, "aids_per_month.png"))
plt.show()


