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

pd.set_option('display.max_columns', None)
con = duckdb.connect(r'D:/CSB_texas/processed/csb_texas - Copy.duckdb')

con.execute("INSTALL spatial;")
con.execute("LOAD spatial;")

# df = con.execute("""SELECT unique_id, platform_name_x AS platform_name, time, depth_mod AS depth,
#                              uncertainty_vert AS uncertainty, uncertainty_hori, Outlier, lat, lon FROM csb_updated WHERE Outlier = 1""").df()

df = con.execute("""SELECT * FROM csb_updated WHERE Outlier = 0""").df()


gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
gdf.set_crs(epsg=4326, inplace=True)
gdf.to_crs(epsg=26914, inplace=True)

# gdf = gpd.GeoDataFrame(df)
#gdf.drop('geom', axis=1, inplace=True)
# Access the column names
column_names = gdf.columns

# Print the column names
print(column_names)
print(gdf.head)
gdf.to_file(r'D:/CSB_texas/processed/csb_cleaned_3.shp')  


# # Define output raster resolution and bounds
# output_resolution = 5  # Adjust as needed
# output_bounds = gdf.total_bounds

# # Create an empty raster
# with rasterio.open(
#     r"D:\CSB_texas\processed\cleaned\combined_CSB_output.tif",
#     "w",
#     driver="GTiff",
#     height=int((output_bounds[3] - output_bounds[1]) / output_resolution),
#     width=int((output_bounds[2] - output_bounds[0]) / output_resolution),
#     count=1,  # Single band output
#     dtype="float32",
#     crs=gdf.crs,
#     transform=rasterio.transform.from_bounds(
#         *output_bounds,
#         int((output_bounds[2] - output_bounds[0]) / output_resolution),
#         int((output_bounds[3] - output_bounds[1]) / output_resolution),
#     ),
# ) as dst:
#     # Burn vector data into the raster
#     rasterize(
#         gdf.geometry,
#         out_shape=dst.shape,
#         fill=1000000,  # Background value
#         out=dst.read(1),
#         transform=dst.transform,
#     )