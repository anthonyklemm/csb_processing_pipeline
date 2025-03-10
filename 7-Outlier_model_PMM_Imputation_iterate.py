# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:04:40 2025

@author: Anthony.R.Klemm
"""

import os
from sklearn.experimental import enable_iterative_imputer  # this enables the sklearn experimental feature
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from scipy.ndimage import uniform_filter1d
import rasterio
from rasterio.transform import from_origin

def detect_outliers(data, scaler, threshold_percentile, original_gdf, return_smoothed=False):
    # Normalize data
    data_scaled = scaler.fit_transform(data)

    # Use MICE algorithm for Predictive Mean Matching Imputation
    imputer = IterativeImputer(
        estimator=LinearRegression(),
        max_iter=15,
        random_state=42,
        sample_posterior=False
    )
    
    # Apply imputation
    data_imputed = imputer.fit_transform(data_scaled)
    
    # Smooth the imputed depth values
    smoothed_depth = uniform_filter1d(data_imputed[:, 2], size=50)
    
    # Calculate residuals and detect outliers
    residuals = np.abs(data_scaled[:, 2] - smoothed_depth)
    threshold = np.percentile(residuals, threshold_percentile)
    outliers = residuals > threshold
    
    # Denormalize smoothed imputed depth
    smoothed_depth_denorm = smoothed_depth * scaler.scale_[2] + scaler.mean_[2]
    
    # Update the original GeoDataFrame's `Outlier` column
    original_gdf.loc[data.index[outliers], 'Outlier'] = True

    # Count the number of outliers
    outlier_count = np.sum(outliers)

    if return_smoothed:
        # Create a full-length smoothed depth array, filling with NaN for removed rows
        full_smoothed_depth = pd.Series(index=original_gdf.index, dtype=float)
        full_smoothed_depth[data.index] = smoothed_depth_denorm
        return full_smoothed_depth, outlier_count

    # Remove outliers from the current dataset for the next iteration
    return data[~outliers], outlier_count

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


# Update the main processing function
def process_geopackages(input_dir, output_dir, plot_dir):
    # Get all GeoPackage files in the input directory
    gpkg_files = [f for f in os.listdir(input_dir) if f.endswith('.gpkg')]
    
    for gpkg_file in gpkg_files:
        print(f"Processing {gpkg_file}...")

        # Read the GeoPackage
        input_path = os.path.join(input_dir, gpkg_file)
        gdf = gpd.read_file(input_path)
        
        # Ensure required columns exist
        if not {'lat', 'lon', 'depth'}.issubset(gdf.columns):
            print(f"Skipping {gpkg_file}: Missing required columns.")
            continue

        # Initialize `Outlier` column to False
        gdf['Outlier'] = False

        # Select columns and normalize data
        processed_data = gdf[["lat", "lon", "depth"]]
        scaler = StandardScaler()

        try:
            # Pass 1: Lenient threshold (99th percentile)
            print("First Pass (Lenient Threshold - 99th percentile):")
            filtered_data_1, outlier_count_1 = detect_outliers(
                processed_data.copy(), scaler, threshold_percentile=99, original_gdf=gdf
            )
            print(f"Outliers detected in Pass 1: {outlier_count_1}")

            # Pass 2: Moderate threshold (98th percentile)
            print("Second Pass (Moderate Threshold - 98th percentile):")
            filtered_data_2, outlier_count_2 = detect_outliers(
                filtered_data_1.copy(), scaler, threshold_percentile=98, original_gdf=gdf
            )
            print(f"Outliers detected in Pass 2: {outlier_count_2}")

            # Pass 3: Final threshold (98th percentile, return smoothed depth)
            print("Third Pass (Strict Threshold - 98th percentile):")
            final_smoothed_depth, outlier_count_3 = detect_outliers(
                filtered_data_2.copy(), scaler, threshold_percentile=98, original_gdf=gdf, return_smoothed=True
            )
            print(f"Outliers detected in Pass 3: {outlier_count_3}")

            # Assign the final smoothed depth back to the GeoDataFrame
            gdf['Final_Smoothed_Depth'] = final_smoothed_depth

            # Save the processed GeoDataFrame to a new GeoPackage
            output_path = os.path.join(output_dir, f"Processed_{gpkg_file}")
            gdf.to_file(output_path, driver='GPKG')
            print(f"Saved processed file to {output_path}")

            # Save the final plot
            plot_path = os.path.join(plot_dir, f"Plot_{gpkg_file.replace('.gpkg', '.png')}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.scatter(
                gdf[~gdf['Outlier']].index, gdf[~gdf['Outlier']]['depth'],
                color='blue', s=1, label="Valid"
            )
            ax.scatter(
                gdf[gdf['Outlier']].index, gdf[gdf['Outlier']]['depth'],
                color='red', s=10, label="Outlier"
            )
            ax.plot(
                gdf.index, gdf['Final_Smoothed_Depth'],
                color='green', linewidth=1.5, label="Final Smoothed Depth"
            )
            ax.set_title(f"Final Outlier Detection for {gpkg_file}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Depth")
            ax.legend()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot to {plot_path}")
            
            # Drop the final smoothed depth from the GeoDataFrame before gpkg export
            gdf = gdf.drop(['Final_Smoothed_Depth'])
            
            # Save the processed GeoDataFrame to a new GeoPackage
            output_path = os.path.join(output_dir, f"Processed_{gpkg_file}")
            gdf.to_file(output_path, driver='GPKG')
            print(f"Saved processed file to {output_path}")
            
        except Exception as e:
            print(f"Error processing {gpkg_file}: {e}")
            
def create_geotiffs_from_cleaned_geopackages(input_dir, output_dir, geotiff_dir, resolution=8):
    """
    Iterates through cleaned GeoPackages and generates GeoTIFFs using only the points where Outlier == False.

    Args:
        input_dir (str): Directory containing cleaned GeoPackages.
        output_dir (str): Directory where GeoTIFFs will be saved.
        geotiff_dir (str): Output directory for GeoTIFFs.
        resolution (int): Resolution of the output GeoTIFFs (default is 8 meters).
    """
    # Get all cleaned GeoPackage files in the output directory
    gpkg_files = [f for f in os.listdir(output_dir) if f.startswith('Processed_') and f.endswith('.gpkg')]

    # Create GeoTIFF output directory if it doesn't exist
    os.makedirs(geotiff_dir, exist_ok=True)

    for gpkg_file in gpkg_files:
        print(f"Generating GeoTIFF for {gpkg_file}...")

        try:
            # Read the cleaned GeoPackage
            cleaned_path = os.path.join(output_dir, gpkg_file)
            gdf = gpd.read_file(cleaned_path)

            # Filter to only include non-outlier points
            cleaned_gdf = gdf[gdf['Outlier'] == False]

            # Define output GeoTIFF filename
            geotiff_filename = os.path.join(geotiff_dir, f"{gpkg_file.replace('.gpkg', '.tif')}")

            # Generate GeoTIFF
            create_geotiff(cleaned_gdf, geotiff_filename, resolution=resolution)
            print(f"Saved GeoTIFF to {geotiff_filename}")

        except Exception as e:
            print(f"Error generating GeoTIFF for {gpkg_file}: {e}")


# Specify the GeoTIFF output directory
geotiff_directory = r'D:/CSB_texas/processed/cleaned'

# Specify the input, output, and plot directories
input_directory = r"D:\CSB_texas\processed\processed_exports"
output_directory = r"D:\CSB_texas\processed\cleaned"
plot_directory = r"D:\CSB_texas\processed\cleaned"


# Ensure output directories exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(plot_directory, exist_ok=True)

# Process all GeoPackages in the input directory
process_geopackages(input_directory, output_directory, plot_directory)
# Run GeoTIFF generation
# non_outlier_gdf = group[group['Outlier'] == False]
# create_geotiff(non_outlier_gdf, tiff_filename)
create_geotiffs_from_cleaned_geopackages(input_directory, output_directory, geotiff_directory, resolution=8)

