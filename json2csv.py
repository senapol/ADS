import json
import csv

# Load the JSON file
with open("/Users/hyoyeon/Desktop/UNI/Year 3/Applied Data Science/ADS/data/nn_frontline_data.json", "r") as f:
    data = json.load(f)

# Open a CSV file for writing
with open("frontline_nodes.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Date", "Longitude", "Latitude"])

    # Go through each date
    for date, polygon_list in data.items():
        for polygon in polygon_list:
            for point in polygon:
                longitude, latitude = point
                writer.writerow([date, longitude, latitude])
