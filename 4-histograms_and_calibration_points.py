# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:38:27 2024

@author: Anthony.R.Klemm
"""

import duckdb
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Connect to DuckDB
con = duckdb.connect(r"E:\csb\kuskokwim\processed\csb_kuskokwim.duckdb")
export_dir = r'E:\csb\kuskokwim\processed\histograms'
os.makedirs(export_dir, exist_ok=True)

# Drop the incorrect 'diff' column if it exists
columns_query = "DESCRIBE csb"
columns_df = con.execute(columns_query).fetchdf()
if 'diff' in columns_df['column_name'].values:
    print("Dropping incorrect 'diff' column in DuckDB...")
    con.execute("ALTER TABLE csb DROP COLUMN diff")

# Check if the 'diff' column exists and create it with the correct calculation
columns_df = con.execute(columns_query).fetchdf()
if 'diff' not in columns_df['column_name'].values:
    print("Creating 'diff' column in DuckDB with correct calculation...")
    con.execute("ALTER TABLE csb ADD COLUMN diff DOUBLE DEFAULT NULL")
    con.execute("UPDATE csb SET diff = (depth_new *-1 - Raster_Value)*-1 WHERE diff IS NULL")

# Fetch all unique identifiers
unique_ids_query = "SELECT DISTINCT unique_id FROM csb"
unique_ids = con.execute(unique_ids_query).fetchdf()['unique_id']

for unique_id in unique_ids:
    print(f'Calculating offset histogram for {unique_id}')
    # Fetch data for the current unique_id
    data_query = f"""
    SELECT unique_id, platform_name_x, diff, lon, lat
    FROM csb
    WHERE unique_id = '{unique_id}' AND diff > -12 AND diff < 12 AND Raster_Value > -20 AND Uncertainty_Value < 3
    """
    data_df = con.execute(data_query).fetchdf()
    
    if not data_df.empty:
        platform_name = data_df['platform_name_x'].iloc[0]  # Assuming platform_name_x is consistent within unique_id
    
        output_csv_path = f'{export_dir}/{unique_id}_csb_offset_analysis.csv'
        data_df.to_csv(output_csv_path)
    
        # Plotting the histogram and density plot for 'diff'
        plt.figure(figsize=(10, 6))
        sns.histplot(data_df['diff'], bins=30, kde=True, color="skyblue", label='Histogram')
        plt.axvline(data_df['diff'].mean(), color='green', linestyle='--', label=f'Mean: {data_df["diff"].mean():.2f}')
        # Draw vertical lines for the standard deviation

        plt.axvline(data_df['diff'].mean() - data_df['diff'].std(), color='purple', linestyle='--', label=f'-1 Std Dev: {(data_df["diff"].mean() - data_df["diff"].std()):.2f}')
        plt.axvline(data_df['diff'].mean() + data_df['diff'].std(), color='purple', linestyle='--', label=f'+1 Std Dev: {(data_df["diff"].mean() + data_df["diff"].std()):.2f}')

        plt.text(data_df['diff'].mean() - data_df['diff'].std(), plt.ylim()[1] * 0.95, f'-1 SD: {data_df["diff"].std():.2f}', horizontalalignment='right', color='purple')
        plt.text(data_df['diff'].mean() + data_df['diff'].std(), plt.ylim()[1] * 0.95, f'+1 SD: {data_df["diff"].std():.2f}', horizontalalignment='left', color='purple')

        plt.title(f'Distribution of Diff Values for {unique_id} ({platform_name})')
        plt.xlabel('Diff')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'{export_dir}/{unique_id}_histogram.png')
        plt.close()

# Close the DuckDB connection
con.close()

