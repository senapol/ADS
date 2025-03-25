import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv("data/cleaned/aid_categories_monthly.csv")
df['announcement_date'] = pd.to_datetime(df['announcement_date'])

# Categories to plot
categories = ['Humanitarian', 'Military equipment', 'Aviation and drones', 
              'Portable defence system', 'Heavy weapon', 'Financial', 'Uncategorised']

df.loc[:, 'Total aid'] = df[categories].sum(axis=1)

# Create the plot
plt.figure(figsize=(14, 6))

# Plot each category as a separate line
# for category in categories:
#     plt.plot(df['announcement_date'], df[category]/1000000, 
#                 marker='o', 
#                 label=category,
#                 linewidth=2,
#                 markersize=8)

plt.plot(df['announcement_date'], df["Total aid"]/1000000, 
            marker='o', 
            label="Total Aid",
            linewidth=2,
            markersize=8)

# Customize the plot
plt.xlabel('Date')
plt.ylabel('Aid Amount (Millions)')
plt.title('Total Monthly Aid')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Use logarithmic scale for y-axis to better handle large variations and zeros
plt.yscale('symlog')
plt.ylim(10**1, 10**5)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure as a PNG file with a dpi of 300
plt.savefig("total_aid_graph.png", dpi=300)

plt.show()