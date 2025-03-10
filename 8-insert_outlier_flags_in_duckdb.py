# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:41:27 2025

@author: Anthony.R.Klemm
"""

import duckdb
import geopandas as gpd
import pandas as pd
import hashlib
import os

conn = duckdb.connect(r'D:/CSB_texas/processed/csb_texas - Copy.duckdb')
geo_package_dir = "D:/CSB_texas/processed/cleaned/"


# Function to create synthetic key
def create_synthetic_key(timestamp, lat, lon):
    composite_string = f"{timestamp}_{lat:.6f}_{lon:.6f}"
    return hashlib.md5(composite_string.encode()).hexdigest()

# Function to process a single GeoPackage and extract outlier synthetic keys
def process_geopackage(geo_package_path):
    try:
        gdf = gpd.read_file(geo_package_path)
        gdf_outliers = gdf[gdf['Outlier'] == True].copy()
        gdf_outliers['time'] = pd.to_datetime(gdf_outliers['time']).astype('int64') // 10**9
        gdf_outliers['lat'] = gdf_outliers['lat'].astype(float)
        gdf_outliers['lon'] = gdf_outliers['lon'].astype(float)
        gdf_outliers['depth'] = gdf_outliers['depth'].astype(float)

        gdf_outliers['synthetic_key'] = gdf_outliers.apply(
            lambda row: create_synthetic_key(row['time'], row['lat'], row['lon']), axis=1
        )
        return gdf_outliers['synthetic_key'].tolist()
    except Exception as e:
        print(f"Error processing GeoPackage {geo_package_path}: {e}")
        return []

# Connect to DuckDB
print("Connecting to DuckDB...")

try:
    # Add Outlier column (if it doesn't exist)
    print("adding Outlier column to the duckdb csb table")
    add_outlier_column = """
    ALTER TABLE csb ADD COLUMN Outlier BOOLEAN DEFAULT FALSE;
    """
    conn.execute(add_outlier_column)
except Exception:
    print("outlier column already exists")
    
    
# Fetch DuckDB data
print("Fetching data from DuckDB...")
duckdb_query = """
SELECT time AS timestamp, lat AS latitude, lon AS longitude, depth_mod AS depth, Outlier
FROM csb
"""
duckdb_df = conn.execute(duckdb_query).fetchdf()

# Standardize DuckDB data
print("Standardizing DuckDB data formats...")
duckdb_df['timestamp'] = pd.to_datetime(duckdb_df['timestamp']).astype('int64') // 10**9
duckdb_df['latitude'] = duckdb_df['latitude'].astype(float)
duckdb_df['longitude'] = duckdb_df['longitude'].astype(float)
duckdb_df['depth'] = duckdb_df['depth'].astype(float)
duckdb_df['synthetic_key'] = duckdb_df.apply(
    lambda row: create_synthetic_key(row['timestamp'], row['latitude'], row['longitude']), axis=1
)

# Debug: Print sample synthetic keys
print("Sample synthetic keys from DuckDB:", duckdb_df['synthetic_key'].head().tolist())

# Process GeoPackages
geo_packages = [os.path.join(geo_package_dir, f) for f in os.listdir(geo_package_dir) if f.endswith('.gpkg')]

all_outlier_keys = set()

print(f"Processing {len(geo_packages)} GeoPackages...")
for geo_package_path in geo_packages:
    print(f"Processing GeoPackage: {geo_package_path}...")
    outlier_keys = process_geopackage(geo_package_path)
    all_outlier_keys.update(outlier_keys)

# Debug: Print sample synthetic keys from GeoPackages
print("Sample synthetic keys from GeoPackages:", list(all_outlier_keys)[:5])

# Count matching keys
matching_keys = duckdb_df['synthetic_key'].isin(all_outlier_keys).sum()
print(f"Number of matching synthetic keys in `csb`: {matching_keys}")

# Update Outlier column
print("Updating DuckDB Outlier column...")
duckdb_df['Outlier'] = duckdb_df['synthetic_key'].isin(all_outlier_keys)

print(duckdb_df)
column_names = duckdb_df.columns

# Print the column names
print(column_names)
print(duckdb_df.head)
# Write updated data back to DuckDB
conn.execute("DROP TABLE IF EXISTS csb_updated;")
conn.execute("""
CREATE TABLE csb_updated AS 
SELECT 
    timestamp, latitude AS lat, longitude AS lon, depth, Outlier
FROM duckdb_df
""")
print("Updated Outlier column written back to DuckDB.")
conn.close()

print("Process complete.")
