# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:38:14 2024

@author: Anthony.R.Klemm
"""

import os
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd
pd.set_option('display.max_columns', None)

def create_polylines(gdf):
    polylines = []
    for unique_id, group in gdf.groupby('unique_id'):
        group['time'] = pd.to_datetime(group['time'])
        group = group.sort_values('time')
        current_line = []
        last_time = None
        for index, row in group.iterrows():
            if last_time is not None:
                time_diff = (row['time'] - last_time).total_seconds() / 60
                if time_diff > 10:
                    if len(current_line) > 1:
                        polylines.append((unique_id, LineString(current_line)))
                    current_line = [row['geometry']]
                else:
                    current_line.append(row['geometry'])
            else:
                current_line.append(row['geometry'])
            last_time = row['time']
        if len(current_line) > 1:
            polylines.append((unique_id, LineString(current_line)))

    lines_gdf = gpd.GeoDataFrame(polylines, columns=['unique_id', 'geometry'], geometry='geometry')
    return lines_gdf

def calculate_distances(lines_gdf):
    distances = lines_gdf.geometry.length
    return distances

def calculate_contributions_and_save_polylines(directory, output_csv, output_shapefile):
    all_polylines = []
    measurement_counts = pd.Series(dtype=int)  # Initialize an empty Series to hold measurement counts
    platform_name = {}  # Dictionary to map unique_id to platform_name

    # Fetch all geopackages
    geopackages = [f for f in os.listdir(directory) if f.endswith('.gpkg')]
    total_files = len(geopackages)  # Total number of geopackages

    for count, filename in enumerate(geopackages, start=1):
        filepath = os.path.join(directory, filename)
        print(f"Processing file {count} of {total_files}: {filepath}")
        
        gdf = gpd.read_file(filepath)

        # Correct CRS check and setting
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

        if 'platform_name' in gdf.columns:
            platform_name.update(gdf.set_index('unique_id')['platform_name'].to_dict())
        else:
            print(f"'platform_name' column not found in {filepath}. Skipping platform name mapping for this file.")

        # Updating measurement counts
        measurement_counts = measurement_counts.add(gdf['unique_id'].value_counts(), fill_value=0)

        gdf = gdf.to_crs(epsg=32618)
        lines_gdf = create_polylines(gdf)
        lines_gdf['Total Distance (meters)'] = calculate_distances(lines_gdf)
        lines_gdf['Platform Name'] = lines_gdf['unique_id'].map(platform_name)
        
        # Append to all_polylines for saving later
        for index, row in lines_gdf.iterrows():
            all_polylines.append((row['unique_id'], row['geometry'], row['Total Distance (meters)'], row['Platform Name']))

    # Convert to DataFrame and GeoDataFrame for distances, platform names, and geometries
    polylines_df = gpd.GeoDataFrame(all_polylines, columns=['unique_id', 'geometry', 'Total Distance (meters)', 'Platform Name'], crs="EPSG:32618")

    # Convert measurement counts to DataFrame and merge
    measurements_df = measurement_counts.reset_index(name='Contributed Measurements').rename(columns={'index': 'unique_id'})
    
    print("polylines_df columns:", polylines_df.columns)
    print("measurements_df columns:", measurements_df.columns)
    
    leaderboard_df = polylines_df.drop(columns='geometry').merge(measurements_df, on='unique_id')

    leaderboard_df = leaderboard_df.groupby(['unique_id', 'Platform Name']).agg({
        'Total Distance (meters)': 'sum',
        'Contributed Measurements': 'first'
    }).reset_index().sort_values('Total Distance (meters)', ascending=False)
    leaderboard_df['Total Distance (meters)'] = leaderboard_df['Total Distance (meters)'] / 1852
    leaderboard_df.rename(columns={'Total Distance (meters)': 'Total Distance (nautical miles)'}, inplace=True)
    leaderboard_df.to_csv(output_csv, index=False)
    polylines_df.drop(columns=['Total Distance (meters)']).to_file(output_shapefile, driver='ESRI Shapefile')

    return leaderboard_df

geopackage_directory = r'D:\CSB_texas\processed\processed_exports'
output_leaderboard_csv = r'D:\CSB_texas\processed\processed_exports\leaderboard.csv'
output_tracklines_shapefile = r'D:\CSB_texas\processed\processed_exports\tracklines.shp'
leaderboard_df = calculate_contributions_and_save_polylines(geopackage_directory, output_leaderboard_csv, output_tracklines_shapefile)
print(leaderboard_df.head())  # Look at the head of the leaderboard_df
