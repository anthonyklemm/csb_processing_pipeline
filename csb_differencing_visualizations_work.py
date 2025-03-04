# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:23:35 2025

@author: Anthony.R.Klemm
"""

import duckdb
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box
import matplotlib.patches as mpatches

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
    provided bounding box (in WGS84) and that are not flagged as outliers.
    Assumes the table contains 'lon', 'lat', 'depth', and 'Outlier' columns.
    Adjust the casting/Outlier filter based on your table's schema.
    """
    con = duckdb.connect(database=db_path)
    query = f"""
    SELECT *
    FROM csb
    WHERE CAST(lon AS DOUBLE) BETWEEN {bbox['min_lon']} AND {bbox['max_lon']}
      AND CAST(lat AS DOUBLE) BETWEEN {bbox['min_lat']} AND {bbox['max_lat']}
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
    The sampled value is added as a new column 'bluetopo_value'.
    """
    # Reproject to the BlueTopo tile's CRS.
    gdf_proj = gdf.to_crs(target_crs)
    
    with rasterio.open(bluetopo_path) as src:
        # Prepare a list of (x, y) coordinates from the reprojected geometries.
        coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]
        sampled_values = [val[0] for val in src.sample(coords)]
    
    # Add the sampled BlueTopo values back to the original GeoDataFrame.
    gdf['bluetopo_value'] = sampled_values
    # Compute discrepancy: CSB depth minus BlueTopo value.
    gdf['discrepancy'] = gdf['depth_mod'] - gdf['bluetopo_value']
    return gdf

def plot_hist2d(gdf):
    """
    Plots a 2D histogram comparing CSB depth_mod vs. BlueTopo values.
    A red dashed line indicates the 1:1 line.
    """
    # import matplotlib.pyplot as plt
    # kwargs = dict(range=[(0,10), (0,10)])
    # import matplotlib.pyplot as plt
    # plt.hist2d(x,y, bins=(100,100), cmin=1, cmap='viridis_r', **kwargs)
    # plt.show()
    # kwargs = dict(range=[(-17,-2), (-17,-2)])
    plt.figure(figsize=(10,10))
    # plt.hist2d(gdf['depth_mod'], gdf['bluetopo_value'], bins=(70,70), cmin=1, cmap='viridis_r', **kwargs)
    plt.hist2d(gdf['depth_mod'], gdf['bluetopo_value'], bins=(70,70), cmin=1, cmap='viridis_r')
    plt.xlabel('CSB depth')
    plt.ylabel('BlueTopo value')
    plt.title('2D Histogram: CSB depth vs. BlueTopo value')
    plt.colorbar(label='Counts')
    # Plot a 1:1 reference line
    min_val = min(gdf['depth_mod'].min(), gdf['bluetopo_value'].min())
    max_val = max(gdf['depth_mod'].max(), gdf['bluetopo_value'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line')
    plt.legend()
    plt.show()

#############################################
# Functions for the difference grid plotting
#############################################

def create_difference_grid(gdf_proj, bounds, cell_size):
    """
    Creates a grid (GeoDataFrame of polygons) over the provided bounds 
    (a tuple: xmin, ymin, xmax, ymax) using the specified cell_size (in the same units as bounds).
    """
    xmin, ymin, xmax, ymax = bounds
    grid_cells = []
    x_coords = np.arange(xmin, xmax, cell_size)
    y_coords = np.arange(ymin, ymax, cell_size)
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf_proj.crs)
    return grid

def aggregate_difference_in_grid(gdf_proj, grid):
    """
    Performs a spatial join between the reprojected points and the grid, then computes
    the mean discrepancy (CSB depth - BlueTopo) for each grid cell.
    """
    # Use the spatial join predicate "within"
    joined = gpd.sjoin(gdf_proj, grid, how='left', predicate='within')
    # Group by grid cell index (the 'index_right' added by sjoin) and compute the mean discrepancy.
    agg = joined.groupby('index_right').agg(mean_diff=('discrepancy', 'mean')).reset_index()
    grid = grid.reset_index().rename(columns={'index': 'grid_index'})
    grid = grid.merge(agg, left_on='grid_index', right_on='index_right', how='left')
    return grid



def plot_difference_grid(grid, extent):
    """
    Bins the absolute mean discrepancy for each grid cell into discrete intervals
    and plots the grid cells colored accordingly, then adds a legend showing the color mapping.
    
    Binning scheme (absolute difference):
        0   to 1.0 m     : dark green (#006400)
        1.0 to 1.5 m     : light green/yellow (#ADFF2F)
        1.5 to 2.0 m     : orange (#FFA500)
        >=2.0 m          : red (#FF0000)
        No data          : light gray (#D3D3D3)
    
    The function removes gridlines by not plotting boundaries and sets the plot extent
    to the provided bounds.
    """
    # Compute the absolute mean discrepancy.
    grid['abs_diff'] = grid['mean_diff'].abs()
    
    # Define bins and associated colors.
    bins = [0, 1, 1.5, 2, np.inf]
    colors = ['#006400', '#ADFF2F', '#FFA500', '#FF0000']
    
    # Bin the absolute difference values.
    grid['color'] = pd.cut(grid['abs_diff'], bins=bins, labels=colors, include_lowest=True)
    # Fill cells with no data with light gray.
    grid['color'] = grid['color'].cat.add_categories(['#D3D3D3']).fillna('#D3D3D3')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot the grid polygons without boundaries so no gridlines appear.
    grid.plot(ax=ax, color=grid['color'], edgecolor=None, alpha=0.7)
    
    # Set the plot extent to the provided bounds (zoom into the data extent)
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    ax.set_title("Difference Grid: Mean Discrepancy (CSB depth_mod - BlueTopo)")
    
    # Create custom legend patches for the defined bins.
    legend_patches = [
        mpatches.Patch(color='#006400', label='0 - 1.0 m'),
        mpatches.Patch(color='#ADFF2F', label='1.0 - 1.5 m'),
        mpatches.Patch(color='#FFA500', label='1.5 - 2.0 m'),
        mpatches.Patch(color='#FF0000', label='>= 2.0 m'),
        mpatches.Patch(color='#D3D3D3', label='No Data')
    ]
    # Place the legend outside the plot area (to the right)
    ax.legend(handles=legend_patches, title="Mean Discrepancy", loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()

def plot_difference_grid_from_points(gdf, tile_crs, cell_size=15):
    """
    Creates and plots a difference grid from the CSB points with discrepancy.
    
    Instead of using the full BlueTopo tile extent, this version calculates the bounding
    box from the CSB data (after reprojecting to the tile's CRS) to "zoom in" on the data.
    
    Parameters:
        gdf        : GeoDataFrame of CSB points (with 'discrepancy') in EPSG:4326.
        tile_crs   : CRS of the BlueTopo tile.
        cell_size  : Grid cell size in the same units as the tile's CRS (e.g., meters).
    """
    # Reproject the points to the tile's CRS.
    gdf_proj = gdf.to_crs(tile_crs)
    
    # Use the bounding box of the CSB points (in tile CRS) as the grid extent.
    zoom_bounds = gdf_proj.total_bounds  # (xmin, ymin, xmax, ymax)
    
    # Create grid covering the extent of the CSB points.
    grid = create_difference_grid(gdf_proj, zoom_bounds, cell_size)
    
    # Aggregate the discrepancy values into each grid cell.
    grid = aggregate_difference_in_grid(gdf_proj, grid)
    
    # Plot the grid with the color-coded bins.
    plot_difference_grid(grid, zoom_bounds)

#############################################
# Example usage in main()
#############################################

def main():
    # For this example, paths are hard-coded.
    bluetopo_path = r"E:\csb\alaska\processed\BlueTopo_VRT\merged_tiles.vrt" # Can be a vrt or standalone bathy surface/geotiff/bag
    db_path = r"E:\csb\alaska\processed\csb_alaska.duckdb"
    
    # (Steps 1-5 from your earlier processing remain the same.)
    # Step 1: Get bounding box (in WGS84) from the BlueTopo tile.
    bbox = get_wgs84_bbox(bluetopo_path)
    print("Bounding Box in WGS84:", bbox)
    
    # Step 2: Query DuckDB for CSB points within this bounding box and not flagged as outliers.
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
    
    # Optionally: Plot a 2D histogram for comparison.
    plot_hist2d(gdf)
    
    # Step 7: Create and plot a higher-resolution difference grid (cell_size=50, for example)
    plot_difference_grid_from_points(gdf, tile_crs, cell_size=15)

if __name__ == "__main__":
    main()
