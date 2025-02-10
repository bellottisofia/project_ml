import geopandas as gpd
import pandas as pd

# Load GeoJSON files
train_df = gpd.read_file('train.geojson')
test_df = gpd.read_file('test.geojson')

# Convert to CSV
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Conversion completed: train.csv and test.csv created successfully!")
