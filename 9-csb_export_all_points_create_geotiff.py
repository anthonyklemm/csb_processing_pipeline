# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:27:06 2025

@author: Anthony.R.Klemm
"""

import duckdb
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import pandas as pd
import os

pd.set_option('display.max_columns', None)

# Connect to DuckDB and load spatial extension
con = duckdb.connect(r'E:\csb\kuskokwim\processed\csb_kuskokwim.duckdb')
con.execute("INSTALL spatial;")
con.execute("LOAD spatial;")

# For example, selecting non-outlier points
df = con.execute("""SELECT * FROM csb WHERE Outlier = 0 AND depth_mod IS NOT NULL""").df()

# If there is a 'geom' column (with WKB data), drop it so we can build our geometry from lat and lon.
if 'geom' in df.columns:
    df = df.drop(columns=['geom'])

# Create a GeoDataFrame from the DuckDB table
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf.set_crs(epsg=4326, inplace=True)
gdf.to_crs(epsg=26903, inplace=True)

# (Optional) Write the GeoDataFrame to a GeoPackage for inspection
output_gpkg = r'E:/csb/kuskokwim/processed/csb_kuskokwim1.gpkg'
os.makedirs(os.path.dirname(output_gpkg), exist_ok=True)
gdf.to_file(output_gpkg, driver='GPKG')
print("GeoPackage successfully written to", output_gpkg)


# Define output raster resolution and bounds
output_resolution = 10  # Adjust as needed
output_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]

# Calculate the raster dimensions
ncols = int((output_bounds[2] - output_bounds[0]) / output_resolution)
nrows = int((output_bounds[3] - output_bounds[1]) / output_resolution)

# Create the affine transform from bounds
transform = rasterio.transform.from_bounds(
    *output_bounds,
    ncols,
    nrows
)

# Define output TIFF path
tiff_output = r'E:/csb/kuskokwim/processed/combined_CSB_output_10m_lzw.tif'
os.makedirs(os.path.dirname(tiff_output), exist_ok=True)

# Create an empty raster with the desired nodata value and burn the depth_mod values.
with rasterio.open(
    tiff_output,
    "w",
    driver="GTiff",
    height=nrows,
    width=ncols,
    compress='lzw',
    count=1,  # Single band output
    dtype="float32",
    crs=gdf.crs,
    transform=transform,
    nodata=1000000,  # Set the nodata value here
) as dst:
    # Burn the vector data into the raster.
    # For each geometry, use its corresponding depth_mod value.
    burned = rasterize(
        ((geom, value) for geom, value in zip(gdf.geometry, gdf['depth_mod'])),
        out_shape=dst.shape,
        fill=1000000,  # Background value (nodata)
        transform=dst.transform,
        dtype="float32"
    )
    dst.write(burned, 1)

print("GeoTIFF created with nodata value 1000000 and depth_mod values burned in.")
