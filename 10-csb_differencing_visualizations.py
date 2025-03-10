# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:09:05 2025

@author: Anthony.R.Klemm
"""

This script:
  1. Gets a bounding box from a BlueTopo tile,
  2. Queries DuckDB for CSB points within that bbox,
  3. Samples the BlueTopo raster at those points and computes a discrepancy,
  4. Builds a grid over the points and aggregates the mean discrepancy,
  5. Plots the classified (color-coded) grid, and
  6. Exports the difference grid as a GeoTIFF (with nodata = 1000000).
"""

import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform_bounds
from rasterio.features import rasterize
from shapely.geometry import Point, box
import matplotlib.patches as mpatches
import os

# ---------------------------
# Helper Functions
# ---------------------------
def get_wgs84_bbox(bluetopo_path):
    """
    Opens a BlueTopo tile, reads its CRS and bounding box,
    and returns the bounding box transformed into WGS84 (EPSG:4326).
    """
    with rasterio.open(bluetopo_path) as src:
        src_crs = src.crs
        src_bounds = src.bounds  # (left, bottom, right, top)
        print("BlueTopo tile native CRS:", src_crs)
        print("Original bounds (native CRS):", src_bounds)
        
        # Transform bounds to WGS84 (EPSG:4326)
        dst_crs = "EPSG:4326"
        bounds_wgs84 = transform_bounds(
            src_crs, dst_crs,
            src_bounds.left, src_bounds.bottom,
            src_bounds.right, src_bounds.top,
            densify_pts=21
        )
        print("Transformed bounds in WGS84:", bounds_wgs84)
        
        return {
            "min_lon": bounds_wgs84[0],
            "min_lat": bounds_wgs84[1],
            "max_lon": bounds_wgs84[2],
            "max_lat": bounds_wgs84[3]
        }

def query_csb_points(db_path, bbox):
    """
    Queries the DuckDB CSB table for points within the
    provided bounding box (in WGS84) that are not flagged as outliers.
    Assumes the table contains 'lon', 'lat', 'depth_mod', and 'Outlier' columns.
    """
    con = duckdb.connect(database=db_path)
    query = f"""
    SELECT *
    FROM csb
    WHERE CAST(lon AS DOUBLE) BETWEEN {bbox['min_lon']} AND {bbox['max_lon']}
      AND CAST(lat AS DOUBLE) BETWEEN {bbox['min_lat']} AND {bbox['max_lat']}
      AND depth_mod IS NOT NULL
      AND Outlier IS FALSE
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df

def create_geodataframe(df):
    """
    Converts a DataFrame with 'lon' and 'lat' columns into a GeoDataFrame
    with CRS EPSG:4326.
    """
    geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf

def sample_bluetopo(gdf, bluetopo_path, target_crs):
    """
    Reprojects the GeoDataFrame from WGS84 to the target CRS (the BlueTopo tile's CRS),
    then uses Rasterio's vectorized sampling to retrieve the BlueTopo value for each point.
    The sampled value is added as a new column 'bluetopo_value' and discrepancy is computed.
    """
    gdf_proj = gdf.to_crs(target_crs)
    with rasterio.open(bluetopo_path) as src:
        coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]
        sampled_values = [val[0] for val in src.sample(coords)]
    gdf['bluetopo_value'] = sampled_values
    gdf['discrepancy'] = gdf['depth_mod'] - gdf['bluetopo_value']
    return gdf

def plot_hist2d(gdf):
    """
    Plots a 2D histogram comparing CSB depth_mod vs. BlueTopo values.
    A red dashed line indicates the 1:1 line.
    """
    plt.figure(figsize=(12,10))
    plt.hist2d(gdf['depth_mod'], gdf['bluetopo_value'], bins=(150,150), cmin=1, cmap='viridis_r')
    plt.xlabel('CSB depth_mod')
    plt.ylabel('BlueTopo value')
    plt.title('2D Histogram: CSB depth_mod vs. BlueTopo value')
    plt.colorbar(label='Counts')
    min_val = min(gdf['depth_mod'].min(), gdf['bluetopo_value'].min())
    max_val = max(gdf['depth_mod'].max(), gdf['bluetopo_value'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.legend()
    plt.show()
# def plot_hist2d(gdf):
#     """
#     Plots a 2D histogram comparing CSB depth_mod vs. BlueTopo values.
#     A red dashed line indicates the 1:1 line.
#     """
#     # Filter out rows with NaN values in 'depth_mod' or 'bluetopo_value'
#     valid = gdf['depth_mod'].notnull() & gdf['bluetopo_value'].notnull()
#     if valid.sum() == 0:
#         print("No valid data available for plotting histogram.")
#         return

#     x = gdf.loc[valid, 'depth_mod']
#     y = gdf.loc[valid, 'bluetopo_value']

#     plt.figure(figsize=(12,10))
#     plt.hist2d(x, y, bins=(150,150), cmin=1, cmap='viridis_r')
#     plt.xlabel('CSB depth')
#     plt.ylabel('BlueTopo value')
#     plt.title('2D Histogram: CSB depth vs. BlueTopo value')
#     plt.colorbar(label='Counts')

#     # Calculate finite min and max values for the 1:1 line
#     min_val = min(x.min(), y.min())
#     max_val = max(x.max(), y.max())
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
#     plt.legend()
#     plt.show()

# ---------------------------
# Difference Grid Functions
# ---------------------------
def create_difference_grid(gdf_proj, bounds, cell_size):
    """
    Creates a grid (GeoDataFrame of polygons) over the provided bounds 
    (a tuple: xmin, ymin, xmax, ymax) using the specified cell_size.
    """
    xmin, ymin, xmax, ymax = bounds
    grid_cells = []
    for x in np.arange(xmin, xmax, cell_size):
        for y in np.arange(ymin, ymax, cell_size):
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf_proj.crs)
    return grid

def aggregate_difference_in_grid(gdf_proj, grid):
    """
    Performs a spatial join between the reprojected points and the grid,
    then computes the mean discrepancy (CSB depth_mod - BlueTopo) for each grid cell.
    """
    joined = gpd.sjoin(gdf_proj, grid, how='left', predicate='within')
    agg = joined.groupby('index_right').agg(mean_diff=('discrepancy', 'mean')).reset_index()
    grid = grid.reset_index().rename(columns={'index': 'grid_index'})
    grid = grid.merge(agg, left_on='grid_index', right_on='index_right', how='left')
    return grid

def plot_difference_grid(grid, extent):
    """
    Bins the absolute mean discrepancy for each grid cell into discrete intervals
    and plots the grid cells colored accordingly, then adds a legend.
    Binning scheme:
        0 to 1.0 m     : dark green (#006400)
        1.0 to 1.5 m   : light green/yellow (#ADFF2F)
        1.5 to 2.0 m   : orange (#FFA500)
        >=2.0 m        : red (#FF0000)
        No data        : light gray (#D3D3D3)
    """
    grid['abs_diff'] = grid['mean_diff'].abs()
    bins = [0, 1, 1.5, 2, np.inf]
    colors = ['#006400', '#ADFF2F', '#FFA500', '#FF0000']
    grid['color'] = pd.cut(grid['abs_diff'], bins=bins, labels=colors, include_lowest=True)
    grid['color'] = grid['color'].cat.add_categories(['#D3D3D3']).fillna('#D3D3D3')
    
    fig, ax = plt.subplots(figsize=(10,8))
    grid.plot(ax=ax, color=grid['color'], edgecolor=None, alpha=0.7)
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("Difference Grid: Mean Discrepancy (CSB depth_mod - BlueTopo)")
    legend_patches = [
        mpatches.Patch(color='#006400', label='0 - 1.0 m'),
        mpatches.Patch(color='#ADFF2F', label='1.0 - 1.5 m'),
        mpatches.Patch(color='#FFA500', label='1.5 - 2.0 m'),
        mpatches.Patch(color='#FF0000', label='>= 2.0 m'),
        mpatches.Patch(color='#D3D3D3', label='No Data')
    ]
    ax.legend(handles=legend_patches, title="Mean Discrepancy", loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def export_difference_grid_to_geotiff(grid, tiff_output, nodata=1000000):
    """
    Exports the aggregated difference grid (with a 'mean_diff' column) to a GeoTIFF.
    Grid cells with no data are assigned the specified nodata value.
    """
    bounds = grid.total_bounds  # [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = bounds
    # Use the width of the first grid cell to determine resolution.
    sample_bounds = grid.geometry.iloc[0].bounds
    cell_width = sample_bounds[2] - sample_bounds[0]
    resolution = cell_width
    ncols = int((xmax - xmin) / resolution)
    nrows = int((ymax - ymin) / resolution)
    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, ncols, nrows)
    
    # Use the 'mean_diff' value for each grid cell; fill missing cells with nodata.
    shapes_gen = ((geom, value) for geom, value in zip(grid.geometry, grid['mean_diff'].fillna(nodata)))
    raster_array = rasterize(
        shapes_gen,
        out_shape=(nrows, ncols),
        transform=transform,
        fill=nodata,
        dtype='float32'
    )
    
    out_meta = {
        'driver': 'GTiff',
        'height': nrows,
        'width': ncols,
        'count': 1,
        'dtype': 'float32',
        'crs': grid.crs.to_string(),
        'transform': transform,
        'nodata': nodata
    }
    os.makedirs(os.path.dirname(tiff_output), exist_ok=True)
    with rasterio.open(tiff_output, "w", **out_meta) as dst:
        dst.write(raster_array, 1)
    print(f"Difference grid GeoTIFF created at {tiff_output}")

# ---------------------------
# Main Processing Function
# ---------------------------
def main():
    # Hard-coded paths (adjust as needed)
    bluetopo_path = r"E:\csb\kuskokwim\processed\Kuskokwim_raw_csb_5m_MLLW.tif"  # Can be a VRT or standalone bathy surface
    db_path = r"E:\csb\kuskokwim\processed\csb_kuskokwim.duckdb"
    
    # Step 1: Get bounding box (in WGS84) from the BlueTopo tile.
    bbox = get_wgs84_bbox(bluetopo_path)
    print("Bounding Box in WGS84:", bbox)
    
    # Step 2: Query DuckDB for CSB points within this bounding box (non-outliers).
    df_csb = query_csb_points(db_path, bbox)
    print("Number of CSB points retrieved:", len(df_csb))
    if df_csb.empty:
        print("No CSB points found in the bounding box. Exiting.")
        return
    
    # Step 3: Convert the DataFrame to a GeoDataFrame.
    gdf = create_geodataframe(df_csb)
    
    # Step 4: Get the BlueTopo tile's CRS.
    with rasterio.open(bluetopo_path) as src:
        tile_crs = src.crs
    
    # Step 5: Sample the BlueTopo raster for each CSB point and compute discrepancy.
    gdf = sample_bluetopo(gdf, bluetopo_path, tile_crs)
    
    # Plot a 2D histogram for visual comparison.
    plot_hist2d(gdf)
    
    # Step 6: Create the difference grid.
    # Reproject the CSB points to the tile's CRS.
    gdf_proj = gdf.to_crs(tile_crs)
    grid = create_difference_grid(gdf_proj, gdf_proj.total_bounds, cell_size=20)
    grid = aggregate_difference_in_grid(gdf_proj, grid)
    
    # Step 6a: Plot the classified difference grid.
    plot_difference_grid(grid, gdf_proj.total_bounds)
    
    # Step 7: Export the difference grid as a GeoTIFF.
    tiff_output = r"E:\csb\kuskokwim\processed\difference_grid1.tif"
    export_difference_grid_to_geotiff(grid, tiff_output, nodata=1000000)

if __name__ == "__main__":
    main()
