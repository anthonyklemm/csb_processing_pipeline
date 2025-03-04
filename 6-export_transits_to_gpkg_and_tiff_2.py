# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:48:51 2025

@author: Anthony.R.Klemm
"""

import os
import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import timedelta
import rasterio
from rasterio.transform import from_origin
import hashlib
import matplotlib.pyplot as plt
import gc
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter1d

def detect_outliers(data, scaler, threshold_percentile, original_gdf, return_smoothed=False):
    # Normalize data - not sure if this is necessary but it works fine when normalized
    data_scaled = scaler.fit_transform(data)
    # Use IterativeImputer (MICE) for imputation
    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=15,
        random_state=42,
        sample_posterior=False
    )
    data_imputed = imputer.fit_transform(data_scaled)
    # Smooth the imputed depth values
    smoothed_depth = uniform_filter1d(data_imputed[:, 2], size=50)
    # Calculate residuals and detect outliers
    residuals = np.abs(data_scaled[:, 2] - smoothed_depth)
    threshold = np.percentile(residuals, threshold_percentile)
    outliers = residuals > threshold
    # Denormalize smoothed imputed depth
    smoothed_depth_denorm = smoothed_depth * scaler.scale_[2] + scaler.mean_[2]
    # Update the original GeoDataFrame's Outlier column
    original_gdf.loc[data.index[outliers], 'Outlier'] = True
    outlier_count = np.sum(outliers)
    if return_smoothed:
        full_smoothed_depth = pd.Series(index=original_gdf.index, dtype=float)
        full_smoothed_depth[data.index] = smoothed_depth_denorm
        return full_smoothed_depth, outlier_count
    return data[~outliers], outlier_count

# --- Helper Functions for Export ---
def create_geotiff(gdf, filename, resolution=8):
    try:
        bounds = gdf.total_bounds
        x_min, y_min, x_max, y_max = bounds
        if x_max == x_min or y_max == y_min:
            raise ValueError("Invalid geographic bounds. All points may be identical or too close.")
        x_res = int((x_max - x_min) / resolution)
        y_res = int((y_max - y_min) / resolution)
        transform = from_origin(x_min, y_max, resolution, resolution)
        out_meta = {
            'driver': 'GTiff',
            'height': y_res,
            'width': x_res,
            'count': 2,
            'dtype': 'float32',
            'crs': gdf.crs.to_string(),
            'transform': transform,
            'nodata': 1000000,
            'compress': 'lzw',
            'interleave': 'band'
        }
        with rasterio.open(filename, "w", **out_meta) as dest:
            for idx, col in enumerate(['depth', 'uncertainty'], start=1):
                array = np.full((y_res, x_res), out_meta['nodata'], dtype='float32')
                for point, value in zip(gdf.geometry, gdf[col]):
                    col_idx = int((point.x - x_min) / resolution)
                    row_idx = int((y_max - point.y) / resolution)
                    if 0 <= col_idx < x_res and 0 <= row_idx < y_res:
                        array[row_idx, col_idx] = value
                dest.write(array, idx)
    except Exception as e:
        print(f"Failed to create GeoTIFF for {filename}. Error: {str(e)}")

def create_transit_ids(df, max_hours_gap, max_days_duration):
    df = df.sort_values(by='time')
    current_transit_id = None
    last_time = None
    current_start_time = None
    transit_ids = []
    for index, row in df.iterrows():
        if last_time is None or (row['time'] - last_time > timedelta(hours=max_hours_gap)) \
           or ((row['time'] - current_start_time) > timedelta(days=max_days_duration)):
            current_transit_id = f"{row['unique_id']}_{row['time'].strftime('%Y-%m-%d_%H-%M-%S')}"
            current_start_time = row['time']
        transit_ids.append(current_transit_id)
        last_time = row['time']
    df['transit_id'] = transit_ids
    return df

def create_synthetic_key(timestamp, lat, lon):
    try:
        timestamp = int(timestamp)
    except Exception:
        timestamp = int(pd.to_datetime(timestamp).timestamp())
    try:
        lat = float(lat)
        lon = float(lon)
    except Exception as e:
        print(f"Error converting lat/lon: {lat}, {lon}")
        raise e
    composite_string = f"{timestamp}_{lat:.6f}_{lon:.6f}"
    return hashlib.md5(composite_string.encode()).hexdigest()

# ---------------------------
# Main Processing Function
# ---------------------------
def main():
    FILE_DIR = r'E:\csb\tampa\processed\exports'
    os.makedirs(FILE_DIR, exist_ok=True)
    MAX_HOURS_GAP = 4
    MAX_DAYS_DURATION = 7
    
    # Connect to DuckDB
    con = duckdb.connect(r'E:\csb\tampa\processed\csb_tampa.duckdb')
    con.load_extension('spatial')
    
    # ------------------------------------------
    # STEP 1: Update csb table with synthetic_key
    # ------------------------------------------
    try:
        con.execute("ALTER TABLE csb ADD COLUMN synthetic_key VARCHAR;")
        print("Added synthetic_key column to csb.")
    except Exception as e:
        print("synthetic_key column already exists or could not be added:", e)
        
    try:
        con.execute("ALTER TABLE csb ADD COLUMN Outlier BOOLEAN DEFAULT FALSE;")
        print("Added Outlier column to csb.")
    except Exception as e:
        print("Outlier column already exists or could not be added:", e)
    
    update_synthetic_key_query = """
    UPDATE csb
    SET synthetic_key = md5(
        cast(EXTRACT(epoch FROM STRPTIME(time, '%Y%m%d %H:%M:%S')) as varchar)
        || '_' || printf('%.6f', CAST(lat as DOUBLE))
        || '_' || printf('%.6f', CAST(lon as DOUBLE))
    )
    """
    con.execute(update_synthetic_key_query)
    print("Updated synthetic_key values in csb.")
    
    # (Optional) Add transit_id column to csb
    try:
        con.execute("ALTER TABLE csb ADD COLUMN transit_id VARCHAR;")
        print("Added transit_id column to csb.")
    except Exception as e:
        print("transit_id column already exists or could not be added:", e)
    
    # ------------------------------------------
    # STEP 2: Process each vessel's data (unique_id)
    # ------------------------------------------
    unique_ids = con.execute("SELECT DISTINCT unique_id FROM csb").fetchall()
    unique_ids = [x[0] for x in unique_ids]
    
    for unique_id in unique_ids:
        query = f"""
        SELECT unique_id, platform_name_x AS platform_name, time, depth_mod AS depth,
               uncertainty_vert AS uncertainty, uncertainty_hori, lat, lon, synthetic_key
        FROM csb
        WHERE unique_id = '{unique_id}'
        AND depth_mod IS NOT NULL
        ORDER BY time;
        """
        df = con.execute(query).df()
        # Convert time column to datetime
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d %H:%M:%S')
        
        # Calculate transit IDs for this vessel subset.
        df = create_transit_ids(df, MAX_HOURS_GAP, MAX_DAYS_DURATION)
        
        # Process each transit group separately.
        for transit_id, group in df.groupby('transit_id'):
            print(f"\nProcessing unique_id {unique_id}, transit {transit_id}")
            # Reset Outlier flag for the group
            group['Outlier'] = False
            
            # --- Run Outlier Detection in three passes ---
            data_for_outlier = group[['lat', 'lon', 'depth']].copy()
            scaler = StandardScaler()
            print("First Pass (99th percentile):")
            filtered_data_1, outlier_count_1 = detect_outliers(data_for_outlier.copy(), scaler, threshold_percentile=99, original_gdf=group)
            print(f"Outliers detected in Pass 1: {outlier_count_1}")
            
            print("Second Pass (98th percentile):")
            filtered_data_2, outlier_count_2 = detect_outliers(filtered_data_1.copy(), scaler, threshold_percentile=98, original_gdf=group)
            print(f"Outliers detected in Pass 2: {outlier_count_2}")
            
            print("Third Pass (98th percentile, strict):")
            final_smoothed_depth, outlier_count_3 = detect_outliers(filtered_data_2.copy(), scaler, threshold_percentile=98, original_gdf=group, return_smoothed=True)
            print(f"Outliers detected in Pass 3: {outlier_count_3}")
            
            # Optionally assign the final smoothed depth
            group['Final_Smoothed_Depth'] = final_smoothed_depth
            
            # --- Save diagnostic outlier plot ---
            # Ensure Outlier column is boolean for plotting.
            group['Outlier'] = group['Outlier'].astype(bool)
            plot_filename = os.path.join(FILE_DIR, f"{unique_id}_{transit_id}_outlier_plot.png")
            fig, ax = plt.subplots(figsize=(12, 6))
            mask_valid = group['Outlier'] == False
            mask_outlier = group['Outlier'] == True
            ax.scatter(group.index[mask_valid], group.loc[mask_valid, 'depth'],
                       color='blue', s=1, label="Valid")
            ax.scatter(group.index[mask_outlier], group.loc[mask_outlier, 'depth'],
                       color='red', s=10, label="Outlier")
            ax.set_title(f"Outlier Detection for Transit {transit_id} (unique_id: {unique_id})")
            ax.set_xlabel("Record Index")
            ax.set_ylabel("Depth")
            ax.legend()
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved outlier plot to {plot_filename}")
            
            # --- Batch update the csb table for this transit group ---
            updates = []
            for idx, row in group.iterrows():
                updates.append((row['synthetic_key'], row['transit_id'], str(row['Outlier']).upper()))
            if updates:
                values_clause = ", ".join(
                    f"('{sk}', '{tid}', {outlier})" for sk, tid, outlier in updates
                )
                bulk_update_query = f"""
                UPDATE csb
                SET transit_id = t.transit_id, Outlier = t.outlier
                FROM (VALUES {values_clause}) AS t(synthetic_key, transit_id, outlier)
                WHERE csb.synthetic_key = t.synthetic_key;
                """
                con.execute(bulk_update_query)
                print("Batch updated csb table for this transit group.")
            # Clear updates list and force garbage collection
            del updates
            gc.collect()
            
            # --- Export this transit group ---
            gdf = gpd.GeoDataFrame(group, geometry=gpd.points_from_xy(group.lon, group.lat))
            gdf.set_crs(epsg=4326, inplace=True)
            # Need to figure out the best way to derive this from the data... not be a user input
            gdf.to_crs(epsg=26917, inplace=True) # Replace this with the EPSG of your desired output (usually NAD83 UTM XXN)
            
            # Export the entire transit group as a GeoPackage.
            start_date = group['time'].min().strftime('%Y%m%d%H%M%S')
            end_date = group['time'].max().strftime('%Y%m%d%H%M%S')
            gpkg_filename = f"{unique_id}_{start_date}_{end_date}.gpkg"
            gpkg_path = os.path.join(FILE_DIR, gpkg_filename)
            gdf.to_file(gpkg_path, driver='GPKG')
            print(f"Exported GeoPackage {gpkg_path}")
            
            # Export one GeoTIFF only from non-outlier points.
            non_outlier_gdf = gdf[gdf['Outlier'] == False]
            if not non_outlier_gdf.empty:
                tiff_filename = gpkg_path.replace('.gpkg', '.tif')
                create_geotiff(non_outlier_gdf, tiff_filename)
                print(f"Exported GeoTIFF {tiff_filename}")
            else:
                print(f"No non-outlier points in transit {transit_id} to export as GeoTIFF.")
            
            # Clear group DataFrame from memory
            del group, gdf
            gc.collect()
    
    con.close()
    print("Process complete.")

if __name__ == "__main__":
    main()
