# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:49:05 2024

@author: Anthony.R.Klemm

This script now supports two options for reference bathymetry:
  - Automated BlueTopo download (which downloads tiles into a 'Modeling' folder,
    then copies them into a separate 'BlueTopo_Tiles' folder and builds a VRT saved in 'BlueTopo_VRT').
  - Alternatively, a local BAG (or GeoTIFF) file can be provided.
"""

import tkinter as tk
from tkinter import filedialog
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import os
from osgeo import gdal, osr
import rasterio
from scipy.interpolate import interp1d
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, LineString
from shapely.validation import make_valid
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import threading
from rasterio.merge import merge
from rasterio.enums import Resampling as rioResampling
from shapely.ops import unary_union
from skimage.morphology import binary_dilation, binary_erosion
import shutil
import glob
import subprocess

# Global variables
title = ""
csb = ""
BAG_filepath = ""
fp_zones = ""
output_dir = ""
resolution = 20  # resolution of quick-look geotiff raster
MASTER_OFFSET_FILE = os.path.join(output_dir, "master_offsets.csv")

pd.set_option('display.max_columns', None)

# ---------------------------
# BAG / BlueTopo functions
# ---------------------------

def loadCSB():
    print('*****Reading CSB input csv file*****')
    df1 = gpd.read_file(csb)  # read in CSB data in CSV
    df1 = df1.astype({'depth': 'float'})
    df2 = df1[(df1['depth'] > 0.5) & (df1['depth'] < 1000)]
    df2 = df2.astype({'time': 'datetime64[ns]'}, errors='ignore')
    df2 = df2.dropna(subset=['time'])
    lower_bound = pd.to_datetime("2014")
    upper_bound = pd.to_datetime(str(datetime.now().year + 1))
    df2 = df2[(df2['time'] > lower_bound) & (df2['time'] < upper_bound)]
    df2 = df2.drop_duplicates(subset=['lon', 'lat', 'depth', 'time', 'unique_id'])
    gdf = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2.lon, df2.lat))
    gdf = gdf.set_crs(4326, allow_override=True)
    return gdf

def reproject_tiff(input_dir, output_dir, target_epsg='EPSG:3395'):
    """
    Searches the input directory for files ending in .tif or .tiff,
    reprojects each to the target EPSG code, and writes the output to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".tif", ".tiff")):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                base_filename, file_extension = os.path.splitext(file)
                new_filename = f"{base_filename}_reprojected{file_extension}"
                output_path = os.path.join(output_dir, relative_path, new_filename)
                output_folder = os.path.dirname(output_path)
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                command = [
                    'gdalwarp', '-t_srs', target_epsg,
                    input_path, output_path
                ]
                try:
                    subprocess.run(command, check=True)
                    # print(f"Reprojected {input_path} to {output_path}")
                except Exception as e:
                    print(f"[DEBUG] Error reprojecting {input_path}: {e}")

def mosaic_tiles(tiles_dir):
    """
    Searches the given tiles directory for reprojected TIFF files (with the "_reprojected" suffix).
    Then it merges the available TIFF files using rasterio.merge.merge, 
    writes the merged raster to a file, and returns the path of the merged raster.
    """
    mosaic_raster_path = os.path.join(tiles_dir, 'merged_tiles.tif')
    
    # Call the reproject_tiff function (make sure it also works with both .tif and .tiff files)
    reproject_tiff(tiles_dir, tiles_dir)
    
    # List to store the opened rasters
    raster_list = []
    print("[DEBUG] Searching for reprojected TIFF files in:", tiles_dir)
    for root, dirs, files in os.walk(tiles_dir):
        for file in files:
            # Use lower-case for a case-insensitive check
            if file.lower().endswith('_reprojected.tif') or file.lower().endswith('_reprojected.tiff'):
                raster_path = os.path.join(root, file)
                print(f"[DEBUG] Found reprojected file: {raster_path}")
                try:
                    src = rasterio.open(raster_path)
                    raster_list.append(src)
                except Exception as e:
                    print(f"[DEBUG] Error opening {raster_path}: {e}")
    
    if not raster_list:
        raise RuntimeError("No input dataset specified. No reprojected TIFF files were found in " + tiles_dir)
    
    print(f"[DEBUG] Total input datasets found: {len(raster_list)}")
    try:
        merged_raster, out_transform = merge(raster_list)
    except Exception as e:
        raise RuntimeError(f"Error during merging: {e}")
    
    # Use metadata from the last opened raster in the list
    out_meta = raster_list[-1].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": merged_raster.shape[1],
        "width": merged_raster.shape[2],
        "transform": out_transform,
        "count": raster_list[-1].count
    })
    
    try:
        with rasterio.open(mosaic_raster_path, "w", **out_meta) as dest:
            for i in range(1, out_meta["count"] + 1):
                dest.write(merged_raster[i - 1, :, :], i)
        print(f"[DEBUG] Merged raster written to {mosaic_raster_path}")
    except Exception as e:
        raise RuntimeError(f"Error writing merged raster: {e}")
    
    # Close all opened raster files
    for src in raster_list:
        src.close()
    
    return mosaic_raster_path


def create_convex_hull_and_download_tiles(csb_data_path, output_dir, use_bluetopo=True):
    """
    Loads CSB data, builds a convex hull, and writes it to a shapefile.
    If use_bluetopo is True, it creates a dedicated Modeling folder,
    downloads BlueTopo tiles, copies them to a separate archive folder,
    and then builds a VRT from all GeoTIFF files found recursively in that folder.
    If use_bluetopo is False, it returns the user‐provided BAG_filepath.
    """

    # Load CSB data and create convex hull
    csb_data = pd.read_csv(csb_data_path)
    gdf = gpd.GeoDataFrame(csb_data, geometry=gpd.points_from_xy(csb_data.lon, csb_data.lat))
    gdf = gdf.set_crs(4326, allow_override=True)
    convex_hull_polygon = gdf.unary_union.convex_hull
    convex_hull_gdf = gpd.GeoDataFrame(geometry=[convex_hull_polygon], crs=gdf.crs)
    convex_hull_shapefile = os.path.join(output_dir, "convex_hull_polygon.shp")
    convex_hull_gdf.to_file(convex_hull_shapefile)
    print(f"Convex hull shapefile written to: {convex_hull_shapefile}")

    if use_bluetopo:
        # Create the 'Modeling' folder for downloading tiles
        bluetopo_tiles_dir = os.path.join(output_dir, "Modeling")
        os.makedirs(bluetopo_tiles_dir, exist_ok=True)

        # Download BlueTopo tiles
        from nbs.bluetopo import fetch_tiles
        fetch_tiles(bluetopo_tiles_dir, convex_hull_shapefile, data_source='modeling')

        # Use the CSV file's title (assumed to be stored in the global variable 'title')
        bluetopo_tiles_copy = os.path.join(output_dir, f"BlueTopo_Tiles_{title}")
        if os.path.exists(bluetopo_tiles_copy):
            shutil.rmtree(bluetopo_tiles_copy)
        shutil.copytree(bluetopo_tiles_dir, bluetopo_tiles_copy)
        print(f"Copied BlueTopo tiles to {bluetopo_tiles_copy}")
        
        # Build a VRT from the copied tiles using a unique folder name that includes the title.
        vrt_dir = os.path.join(output_dir, f"BlueTopo_VRT_{title}")
        os.makedirs(vrt_dir, exist_ok=True)
        
        # Create glob patterns to find both .tif and .tiff files in the unique tiles folder.
        tif_pattern = os.path.join(bluetopo_tiles_copy, '**', '*.tif')
        tiff_pattern = os.path.join(bluetopo_tiles_copy, '**', '*.tiff')
        print(f"[DEBUG] Glob pattern for .tif: {tif_pattern}")
        print(f"[DEBUG] Glob pattern for .tiff: {tiff_pattern}")
        tile_files = glob.glob(tif_pattern, recursive=True) + glob.glob(tiff_pattern, recursive=True)
        print("[DEBUG] Found the following tile files for VRT building:")
        for f in tile_files:
            print("  ", f)
        
        if not tile_files:
            raise RuntimeError("No BlueTopo GeoTIFF files were found in " + bluetopo_tiles_copy)
        
        # Save the VRT file with the title appended to the file name.
        vrt_path = os.path.join(vrt_dir, f"merged_tiles_{title}.vrt")
        vrt = gdal.BuildVRT(vrt_path, tile_files)
        vrt = None  # Close the VRT dataset
        print(f"Created VRT at {vrt_path}")
        return vrt_path
    else:
        # If not using automated download, assume BAG_filepath is provided by the user.
        return BAG_filepath

    
def reproject_raster(input_raster, output_raster, dst_crs):
    with rasterio.open(input_raster) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'count': 1,
            'dtype': src.dtypes[0]
        })

        with rasterio.open(output_raster, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),  # Reproject only the first band
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

def fetch_tide_data(station_id, start_date, end_date, product, interval=None, attempt_great_lakes=False):
    base_url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "begin_date": start_date,
        "end_date": end_date,
        "station": station_id,
        "datum": "MLLW" if not attempt_great_lakes else "LWD",
        "time_zone": "gmt",
        "units": "metric",
        "format": "json",
        "product": product
    }

    if interval:
        params["interval"] = interval

    request_url = requests.Request('GET', base_url, params=params).prepare().url
    print(f"Requesting URL: {request_url}")

    response = requests.get(base_url, params=params)
    data = response.json()

    if 'predictions' in data:
        df = pd.json_normalize(data['predictions'])
        data_type = "predicted data"
    elif 'data' in data:
        df = pd.json_normalize(data['data'])
        data_type = "observed data"
    else:
        print(f"No data returned for URL: {request_url}")
        return pd.DataFrame()

    df['t'] = pd.to_datetime(df['t'])
    # Convert 'v' to numeric, coercing errors to NaN
    df['v'] = pd.to_numeric(df['v'], errors='coerce')
    if df['v'].isna().any():
        print("Warning: Some tide values could not be converted to numeric and will be dropped.")
        df = df.dropna(subset=['v'])

    print(f"Pulled {data_type} for station {station_id} from {start_date} to {end_date}")
    return df

def check_for_gaps(dataframe, max_gap_duration='1h'):
    gaps = dataframe['t'].diff() > pd.Timedelta(max_gap_duration)
    return gaps.any()

def cosine_interpolation(df, start_date, end_date):
    df['time_num'] = (df['t'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    df = df.sort_values('time_num')
    interp_func = interp1d(df['time_num'], df['v'], kind='cubic')
    time_num_grid = np.linspace(df['time_num'].min(), df['time_num'].max(), num=3000)
    v_grid = interp_func(time_num_grid)
    time_grid = pd.to_datetime(time_num_grid, unit='s')
    
    # Trimming
    trim_start = pd.to_datetime(start_date) + pd.Timedelta(hours=12)
    trim_end = pd.to_datetime(end_date) - pd.Timedelta(hours=12)
    trimmed_df = pd.DataFrame({'t': time_grid, 'v': v_grid})
    trimmed_df = trimmed_df[(trimmed_df['t'] >= trim_start) & (trimmed_df['t'] <= trim_end)]
    #print(trimmed_df)
    return trimmed_df

def create_survey_outline(raster_path, output_dir, title, desired_resolution=8, dilation_iterations=3, erosion_iterations=2):
    print("starting create_survey_outline() function")
    with rasterio.open(raster_path) as raster:
        # Resample the raster
        data = raster.read(
            1,  # Reading only the first band
            out_shape=(
                raster.height // desired_resolution,
                raster.width // desired_resolution
            ),
            resampling=Resampling.bilinear
        )

        # Create a binary mask
        nodata = raster.nodatavals[0] or 1000000
        binary_mask = (data != nodata).astype(np.uint8)

        # Apply dilation and erosion
        for _ in range(dilation_iterations):
            binary_mask = binary_dilation(binary_mask)
        for _ in range(erosion_iterations):
            binary_mask = binary_erosion(binary_mask)

        # Ensure binary_mask is of type uint8
        binary_mask = binary_mask.astype(np.uint8)

        # Generate shapes from the binary mask
        transform = raster.transform * raster.transform.scale(
            (raster.width / data.shape[-1]),
            (raster.height / data.shape[-2])
        )

        #polygons = [shape(geom) for geom, val in shapes(binary_mask, mask=binary_mask, transform=transform) if val == 1]

        # Perform unary union
        #unified_geometry = unary_union(polygons)
        # Generate polygons from the binary mask and make them valid
        polygons = [make_valid(shape(geom)) for geom, val in shapes(binary_mask, mask=binary_mask, transform=transform) if val == 1]

        # Simplify polygons to reduce complexity
        simplified_polygons = [polygon.simplify(tolerance=0.001, preserve_topology=True) for polygon in polygons]

        # Perform unary union on simplified, valid polygons
        unified_geometry = unary_union(simplified_polygons)
        # Reproject unified geometry to WGS84 before simplification
        geo_df = gpd.GeoDataFrame(geometry=[unified_geometry], crs=raster.crs)
        geo_df = geo_df.to_crs(epsg=4326)

        # Simplify the geometry
        geo_df['geometry'] = geo_df.geometry.simplify(tolerance=0.001)

        # Check and fix bad topology if necessary
        geo_df['geometry'] = geo_df.geometry.apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

        # Save to a shapefile
        bathy_polygon_shp = f"{output_dir}/{title}_bathy_polygon.shp"
        geo_df.to_file(bathy_polygon_shp, driver='ESRI Shapefile')

        print('Bathymetry polygon shapefile created.')
        return bathy_polygon_shp

def tides():
    gdf = loadCSB()
    print('CSB data from csv file loaded. Starting tide correction')

    zones = gpd.read_file(fp_zones)
    join = gpd.sjoin(gdf, zones, how='inner', predicate='within')
    join = join.astype({'time': 'datetime64[ns]'})
    join = join.sort_values('time')

    def generate_date_ranges(dates):
        dates.sort()
        date_ranges = []
        for date in dates:
            if not date_ranges or date - pd.Timedelta(days=1) > date_ranges[-1][1]:
                date_ranges.append([date, date])
            else:
                date_ranges[-1][1] = date
        date_ranges = [(start_date - pd.Timedelta(days=1), end_date + pd.Timedelta(days=1)) for start_date, end_date in date_ranges]
        return [(start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')) for start_date, end_date in date_ranges]

    tdf = []
    known_subordinate_stations = set()
    known_great_lakes_stations = set()

    for station_id in join['ControlStn'].unique():
        station_dates = join[join['ControlStn'] == station_id]['time'].dt.floor('d').unique()
        date_ranges = generate_date_ranges(list(station_dates))

        for start_date, end_date in date_ranges:
            # Try to fetch observed data first
            verified_data = fetch_tide_data(station_id, start_date, end_date, product="water_level")
            if not verified_data.empty and not check_for_gaps(verified_data):
                tdf.append(verified_data)
            else:
                # If there are gaps, try fetching 6-minute predicted data
                predicted_data = fetch_tide_data(station_id, start_date, end_date, product="predictions")
                if not predicted_data.empty and not check_for_gaps(predicted_data):
                    tdf.append(predicted_data)
                else:
                    # If that doesn't work, fallback to predicted hilo data
                    hilo_predictions = fetch_tide_data(station_id, start_date, end_date, product="predictions", interval='hilo')
                    if not hilo_predictions.empty:
                        interpolated_hilo = cosine_interpolation(hilo_predictions, start_date, end_date)
                        tdf.append(interpolated_hilo)
                        known_subordinate_stations.add(station_id)
                    else:
                        great_lakes_data = fetch_tide_data(station_id, start_date, end_date, product="water_level", attempt_great_lakes=True)
                        if not great_lakes_data.empty:
                            tdf.append(great_lakes_data)
                            known_great_lakes_stations.add(station_id)
                        else:
                            print(f"No water level data available for station {station_id}.")

    if tdf:
        tdf = pd.concat(tdf)
        print("Concatenated tdf shape:", tdf.shape)

        tdf = tdf.sort_values('t')
        jtdf = pd.merge_asof(join, tdf, left_on='time', right_on='t')
        print("jtdf shape before column drop:", jtdf.shape)

        columns_to_drop = ['Shape__Are', 'Shape__Len', 'Input_FID', 'id', 'name', 'state', 'affil', 
                           'latitude', 'longitude', 'data', 'metaapi', 'dataapi', 'Shape_Le_2']
        jtdf.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        print("jtdf shape after column drop:", jtdf.shape)

        jtdf = jtdf.dropna(subset=['depth', 'time', 'geometry'])
        print("jtdf shape after dropna:", jtdf.shape)

        jtdf['t_corr'] = jtdf['t'] + pd.to_timedelta(jtdf['ATCorr'], unit='m')

        newdf = jtdf[['t_corr', 'v']].copy()
        print("newdf shape before dropna:", newdf.shape)

        newdf = newdf.rename(columns={'v': 'v_new', 't_corr': 't_new'})
        newdf = newdf.sort_values('t_new').dropna()
        print("newdf shape after dropna:", newdf.shape)

        csb_corr = pd.merge_asof(jtdf, newdf, left_on='time', right_on='t_new', direction='nearest')
        print("csb_corr shape before dropna:", csb_corr.shape)

        print("csb_corr shape after dropna:", csb_corr.shape)

        csb_corr['depth_new'] = csb_corr['depth'] - (csb_corr['RR'] * csb_corr['v_new'])
        print("csb_corr shape after applying tide corrections:", csb_corr.shape)

        csb_corr = gpd.GeoDataFrame(csb_corr, geometry='geometry', crs='EPSG:4326')
        csb_corr['time'] = csb_corr['time'].dt.strftime("%Y%m%d %H:%M:%S")

        csb_corr = csb_corr[(csb_corr['depth'] > 1.5) & (csb_corr['depth'] < 1000)]
        csb_corr = csb_corr.rename(columns={'depth': 'depth_old'}).drop(columns=['index_right', 'ATCorr', 'RR', 'ATCorr2', 'RR2', 'Shape_Leng', 'Shape_Area', 'Shape_Le_1', 't', 'v', 't_corr', 't_new', 'v_new'])
        return csb_corr
    else:
        print("No tide data available for the specified period.")
        return pd.DataFrame()

def BAGextract():
    print("starting BAGextract() function")
    global BAG_filepath
    BAG_filepath = os.path.abspath(BAG_filepath)  # Ensure it's an absolute path
    print("DEBUG - BAG_filepath:", BAG_filepath)
    print('*****Starting to import BAG bathy and aggregate to 8m geotiff*****')

    # Translate options and creation of multi-band GeoTIFF
    translate_options = gdal.TranslateOptions(bandList=[1, 2], creationOptions=['COMPRESS=LZW'])
    intermediate_raster_path = output_dir + '/' + title + '_intermediate.tif'

    dataset = gdal.Open(BAG_filepath, gdal.GA_ReadOnly)
    if dataset is None:
        print("Error: Unable to open the file. Check the file path.")
        return None, None

    gdal.Translate(intermediate_raster_path, dataset, options=translate_options)

    # Check the CRS
    crs = osr.SpatialReference(wkt=dataset.GetProjection())
    dataset = None  # Close the dataset

    if crs.IsGeographic():
        print("Reprojecting GeoTIFF to a projected CRS...")
        reprojected_raster_path = output_dir + '/' + title + '_reprojected.tif'
        gdal.Warp(reprojected_raster_path, intermediate_raster_path, dstSRS='EPSG:3395')
        intermediate_raster_path = reprojected_raster_path

    # Open the intermediate raster to check its resolution
    with rasterio.open(intermediate_raster_path) as src:
        res_x, res_y = src.res

    # Check if the resolution is coarser than 8m
    if max(res_x, res_y) > 8:
        output_raster = intermediate_raster_path
    else:
        # Resample to desired resolution
        output_raster_resampled = output_dir + '/' + title + '_5m_MLLW.tif'
        gdal.Warp(output_raster_resampled, intermediate_raster_path, xRes=8, yRes=8)
        output_raster = output_raster_resampled
        # Clean up intermediate file
        os.remove(intermediate_raster_path)

    # Replace NaN values and update nodata value in the raster
    with rasterio.open(output_raster) as src:
        data = src.read()
        meta = src.meta

    data = np.where(np.isnan(data), 1000000, data)
    meta.update(nodata=1000000)

    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(data)

    # Reproject the raster to WGS84
    output_raster_wgs84 = output_dir + '/' + title + '_wgs84.tif'
    gdal.Warp(output_raster_wgs84, output_raster, dstSRS='EPSG:4326')

    # Call create_survey_outline to generate the bathymetry polygon shapefile
    bathy_polygon_shp = create_survey_outline(output_raster_wgs84, output_dir, title)

    # Clean up intermediate files
    try:
        os.remove(intermediate_raster_path)
        os.remove(output_raster_resampled)
        os.remove(output_raster_wgs84)
    except Exception:
        pass

    return output_raster_wgs84, bathy_polygon_shp

def get_raster_values(x, y, raster):
    ds = gdal.Open(raster)
    gt = ds.GetGeoTransform()
    nodata = [ds.GetRasterBand(i + 1).GetNoDataValue() for i in range(ds.RasterCount)]

    px = int((x - gt[0]) / gt[1])
    py = int((y - gt[3]) / gt[5])

    if px < 0 or py < 0 or px >= ds.RasterXSize or py >= ds.RasterYSize:
        return [np.nan] * ds.RasterCount

    values = []
    for i in range(ds.RasterCount):
        band = ds.GetRasterBand(i + 1)
        value = band.ReadAsArray(px, py, 1, 1)[0][0]
        values.append(value if value != nodata[i] else np.nan)

    return values

def read_master_offsets():
    """Reads the master offsets from a CSV file."""
    MASTER_OFFSET_FILE = os.path.join(output_dir, "master_offsets.csv")

    if os.path.exists(MASTER_OFFSET_FILE):
        return pd.read_csv(MASTER_OFFSET_FILE)
    else:
        return pd.DataFrame(columns=['unique_id', 'platform_name', 'offset_value', 'std_dev', 'accuracy_score', 'date_range', 'tile_name'])

def update_master_offsets(unique_id, platform_name, new_offset, std_dev, date_range, tile_name):
    global MASTER_OFFSET_FILE
    MASTER_OFFSET_FILE = os.path.join(output_dir, "master_offsets.csv")
    master_offsets = read_master_offsets()


    accuracy_score = 1 / std_dev if std_dev != 0 else 0

    #print('checking for existing offset by unique_id and platform_name')
    existing_index = master_offsets[(master_offsets['unique_id'] == unique_id)].index

    new_row = pd.DataFrame([{
        'unique_id': unique_id,
        'platform_name': platform_name,
        'offset_value': new_offset,
        'std_dev': std_dev,
        'accuracy_score': accuracy_score,
        'date_range': date_range,
        'tile_name': tile_name
    }])

    # Exclude empty or all-NA entries before concatenation
    new_row = new_row.dropna(how='all')

    if existing_index.empty:
        master_offsets = pd.concat([master_offsets, new_row], ignore_index=True)
    else:
        if master_offsets.loc[existing_index[0], 'accuracy_score'] <= accuracy_score:
            master_offsets.loc[existing_index[0], list(new_row.columns)] = new_row.iloc[0]

    try:
        master_offsets.to_csv(MASTER_OFFSET_FILE, index=False)
        #print(f"Master offsets updated successfully in {MASTER_OFFSET_FILE}.")
    except Exception as e:
        print(f"Failed to update master offsets: {e}")

def derive_draft():
    output_raster, raster_boundary_shp = BAGextract()
    csb_corr = tides()

    raster_boundary = gpd.read_file(raster_boundary_shp)
    raster_boundary['geometry'] = raster_boundary['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    csb_corr['geometry'] = csb_corr['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))
    csb_corr['row_id'] = range(len(csb_corr))

    csb_corr_subset = csb_corr[csb_corr.geometry.within(raster_boundary.geometry.unary_union)]
    sampled_csb_corr_subset = pd.DataFrame()
    date_ranges = {}

    for name, group in csb_corr_subset.groupby('unique_id'):
        sampled_group = group.sample(n=1000) if len(group) > 1000 else group
        sampled_csb_corr_subset = pd.concat([sampled_csb_corr_subset, sampled_group], ignore_index=True)
        sampled_group['time'] = pd.to_datetime(sampled_group['time'])
        min_timestamp = sampled_group['time'].min()
        max_timestamp = sampled_group['time'].max()
        date_ranges[name] = (min_timestamp.strftime('%Y%m%d'), max_timestamp.strftime('%Y%m%d')) if not pd.isnull(min_timestamp) and not pd.isnull(max_timestamp) else ("19700101", "19700101")

    try:
        sampled_csb_corr_subset[['Raster_Value', 'Uncertainty_Value']] = pd.DataFrame(
            sampled_csb_corr_subset.apply(
                lambda row: get_raster_values(row.geometry.x, row.geometry.y, output_raster), axis=1
            ).tolist(), index=sampled_csb_corr_subset.index
        )
    except KeyError as e:
        print(f"KeyError encountered during selection of Raster_Value from reference bathy: {e}")
    except Exception as e:
        print(f"Unexpected error encountered during selection of Raster_Value from reference bathy: {e}")

    try:
        csb_corr = csb_corr.merge(sampled_csb_corr_subset[['row_id', 'Raster_Value', 'Uncertainty_Value']], on='row_id', how='left')
    except KeyError as e:
        print(f'KeyError encountered during merging of sampled_csb_corr_subset and csb_corr: {e}')
    except Exception as e:
        print(f'Unexpected error encountered during merging of sampled_csb_corr_subset and csb_corr: {e}')

    filtered_csb_corr = csb_corr[csb_corr['Uncertainty_Value'] < 4]

    try:
        if 'Raster_Value' in filtered_csb_corr.columns and 'depth_new' in filtered_csb_corr.columns:
            filtered_csb_corr['diff'] = filtered_csb_corr['depth_new'] - (filtered_csb_corr['Raster_Value'] * -1)
    except KeyError as e:
        print(f'KeyError encountered calculating csb_corr[diff] field: {e}')
    except Exception as e:
        print(f'Unexpected error encountered calculating csb_corr[diff] field: {e}')

    try:
        if 'diff' in filtered_csb_corr.columns:
            out = filtered_csb_corr.groupby('unique_id')['diff'].agg(['mean', 'std', 'count']).reset_index()
    except KeyError as e:
        print(f'KeyError encountered creating out dataframe: {e}')
    except Exception as e:
        print(f'Unexpected error encountered creating out dataframe: {e}')

    if 'out' in locals():
        out.loc[:, 'mean'] = out['mean'].fillna(0)
        out.loc[:, 'std'] = out['std'].fillna(999)
        out.loc[:, 'count'] = out['count'].fillna(0)
        out.loc[(out['mean'] > 3) | (out['mean'] < -11), ['mean', 'std', 'count']] = [0, 999, 0]
        out.loc[(out['std'] > 7), ['mean', 'std', 'count']] = [0, 999, 0]
        out.to_csv(output_dir + '/VESSEL_OFFSETS_csb_corr_' + title + '.csv', mode='a')

        master_offsets = read_master_offsets()
        platform_mapping = filtered_csb_corr[['unique_id', 'platform_name']].drop_duplicates()
        out_with_platform = out.merge(platform_mapping, on='unique_id', how='left')

        for index, row in out_with_platform.iterrows():
            unique_id = row['unique_id']
            platform_name = row['platform_name']
            new_offset = row['mean']
            std_dev = row['std']
            date_range = date_ranges.get(unique_id, ("19700101", "19700101"))
            update_master_offsets(unique_id, platform_name, new_offset, std_dev, date_range, title)
    else:
        print("Variable 'out' is not defined. Skipping further processing.")

    return csb_corr

def draft_corr():
    csb_corr = derive_draft()
    master_offsets = read_master_offsets()

    # Merge the CSB data with the master offsets based on unique vessel ID
    csb_corr1 = csb_corr.merge(master_offsets, left_on='unique_id', right_on='unique_id', how='left')

    # Apply the offset correction
    csb_corr1['depthfinal'] = csb_corr1['depth_new'] - csb_corr1['offset_value']
    csb_corr1['depthfinal'] = csb_corr1['depthfinal'] * -1  
    
    # try to drop some unneeded columns
    csb_corr1 = csb_corr1.drop(columns=['s', 'f', 'q', 'DataProv', 'ControlS_2', 'ControlS_1', 'row_id', 'platform_name_y'], errors='ignore')
    
    print('*****Exporting tide and vessel offset corrected CSB to geopackage*****')
    csb_corr1.to_file(output_dir + '/csb_processed_'+ title +'.gpkg', driver='GPKG', layer = 'csb')
    
    return(csb_corr1)

def reproject_to_mercator(geodataframe):
    # Reproject to EPSG:3395 (World Mercator)
    return geodataframe.to_crs(epsg=3395)

def rasterize_with_rasterio(geodataframe, output_path, resolution=50, nodatavalue=1000000):
    # Reproject the geodataframe
    geodataframe = reproject_to_mercator(geodataframe)

    # Debugging steps: Check if the GeoDataFrame is empty
    if geodataframe.empty:
        print("Error: The GeoDataFrame is empty.")
        return
    else:
        print(f"Number of features in the GeoDataFrame: {len(geodataframe)}")
        print(geodataframe.head())

    # Define the bounds of your raster
    minx, miny, maxx, maxy = geodataframe.total_bounds
    print("Spatial extent:", minx, miny, maxx, maxy)

    # Check if bounds are reasonable
    if (maxx-minx) <= 0 or (maxy-miny) <= 0:
        print("Error: Invalid spatial extent.")
        return

    # Calculate the dimensions of the raster
    x_res = int((maxx - minx) / resolution)
    y_res = int((maxy - miny) / resolution)

    # Debugging step: Check if resolution is leading to zero dimensions
    if x_res <= 0 or y_res <= 0:
        print("Error: Resolution too high or invalid spatial extent, leading to zero dimensions.")
        return

    # Define the transform
    transform = from_origin(minx, maxy, resolution, resolution)

    # Define the output raster dimensions and CRS
    out_meta = {
        'driver': 'GTiff',
        'height': y_res,
        'width': x_res,
        'count': 1,
        'dtype': 'float32',
        'crs': geodataframe.crs.to_string(),
        'transform': transform,
        'nodata': nodatavalue
    }

    # Rasterize the geometries
    with rasterio.open(output_path, 'w', **out_meta) as out_raster:
        out_raster.write(rasterize(
            ((geom, value) for geom, value in zip(geodataframe.geometry, geodataframe['depthfinal'])),
            out_shape=(y_res, x_res),
            transform=transform,
            fill=nodatavalue
        ), 1)


def plot_vessel_tracks(gdf):
    # Check which platform_name variant exists and use it
    platform_col = None
    for potential_name in ['platform_name', 'platform_name_x', 'platform_name_y']:
        if potential_name in gdf.columns:
            platform_col = potential_name
            break
    if platform_col is None:
        raise ValueError("No platform_name column found in DataFrame.")
    
    # Ensure 'gdf' is in WGS84
    gdf = gdf.to_crs(epsg=4326)
    
    # Ensure 'time' column is in datetime format
    gdf['time'] = pd.to_datetime(gdf['time'])

    # Initialize an empty dictionary for LineStrings with platform names as keys
    lines_by_platform = {}

    # Group by the platform name column found
    for platform_name, platform_group in gdf.groupby(platform_col):
        # Further group each platform's data by 'unique_id'
        for unique_id, group in platform_group.groupby('unique_id'):
            group = group.sort_values('time')
            current_line = []

            for i, row in group.iterrows():
                if len(current_line) > 0:
                    time_diff = row['time'] - current_line[-1][1]
                    if time_diff.total_seconds() > 600:  # 10 minutes
                        if len(current_line) > 1:
                            line = LineString([point for point, _ in current_line])
                            lines_by_platform.setdefault(platform_name, []).append(line)
                        current_line = []
                current_line.append((row.geometry, row['time']))
            
            if len(current_line) > 1:
                line = LineString([point for point, _ in current_line])
                lines_by_platform.setdefault(platform_name, []).append(line)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each platform's lines with a unique color and add to the legend
    for platform_name, lines in lines_by_platform.items():
        for line in lines:
            ax.plot(*line.xy, label=platform_name)
    
    # Handling duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust the visualization
    ax.set_title(title + " CSB Vessel Tracks")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir + '/csb_plot_' + title + '.png', dpi=300)
    plt.close()
    
def rasterize_CSB():
    csb_corr1 = draft_corr()
    print('*****Rasterizing CSB data and exporting geotiff*****')
    output_raster_path = output_dir + '/csb_processed_' + title + '.tif'
    rasterize_with_rasterio(csb_corr1, output_raster_path, resolution)    
    
    # Call the new plot function
    plot_vessel_tracks(csb_corr1)
    
def open_file_dialog(var, file_types):
    filename = filedialog.askopenfilename(filetypes=file_types)
    var.set(filename)

def open_folder_dialog(var):
    foldername = filedialog.askdirectory()
    var.set(foldername)

def process_csb_threaded():
    csb_directory = csb_var.get()
    if not os.path.isdir(csb_directory):
        print("Selected path is not a directory.")
        return
    processing_thread = threading.Thread(target=process_csb_for_directory, args=(csb_directory,))
    processing_thread.start()

def process_csb_for_directory(csb_directory):
    start_time = time.time()
    global csb, title, fp_zones, use_bluetopo, output_dir, BAG_filepath
    csb = csb_var.get()
    fp_zones = fp_zones_var.get()
    output_dir = output_dir_var.get()
    csb_directory = os.path.abspath(csb_directory)
    fp_zones = os.path.abspath(fp_zones)
    output_dir = os.path.abspath(output_dir)
    print(f"output_dir is: {output_dir}")

    csv_files = [os.path.join(csb_directory, f) for f in os.listdir(csb_directory) if f.endswith('.csv')]
    for csb_file in csv_files:
        csb = csb_file
        title = os.path.splitext(os.path.basename(csb_file))[0]
        output_gpkg_path = os.path.join(output_dir, f"csb_processed_{title}.gpkg")
        if os.path.exists(output_gpkg_path):
            print(f"Skipping already processed file: {title}")
            continue
        print(f"Processing {csb} with title: {title}")
        try:
            modeling_dir_path = os.path.join(output_dir, "Modeling")
            try:
                shutil.rmtree(modeling_dir_path)
                print(f"Successfully deleted folder: {modeling_dir_path}")
            except Exception as e:
                print(f"Error deleting folder: {e} - or it didn't exist")
            # If automated BlueTopo download is selected, download tiles and build VRT.
            if bluetopo_var.get():
                BAG_filepath = create_convex_hull_and_download_tiles(csb, output_dir, use_bluetopo=True)
                print(f"Updated BAG_filepath (VRT): {BAG_filepath}")
                # Optionally, you could set a flag or global variable to indicate automated mode.
            else:
                # Use the user-provided BAG_filepath (local input)
                BAG_filepath = BAG_filepath_var.get()
                print(f"Using user-provided BAG_filepath: {BAG_filepath}")
            
            # Proceed with the rest of your processing (tide correction, etc.)
            rasterize_CSB()  # This calls your function that uses BAG_filepath internally
            
            # Clean up the Modeling directory and intermediate files if they exist
            try:
                shutil.rmtree(modeling_dir_path)
                print(f"Successfully deleted folder: {modeling_dir_path}")
                files_to_delete = [
                    os.path.join(output_dir, 'convex_hull_polygon.cpg'),
                    os.path.join(output_dir, 'convex_hull_polygon.shp'),
                    os.path.join(output_dir, 'convex_hull_polygon.dbf'),
                    os.path.join(output_dir, 'convex_hull_polygon.prj'),
                    os.path.join(output_dir, 'convex_hull_polygon.shx'),
                    os.path.join(output_dir, f'{title}_bathy_polygon.cpg'),
                    os.path.join(output_dir, f'{title}_bathy_polygon.shp'),
                    os.path.join(output_dir, f'{title}_bathy_polygon.dbf'),
                    os.path.join(output_dir, f'{title}_bathy_polygon.shx'),
                    os.path.join(output_dir, f'{title}_bathy_polygon.prj'),
                    os.path.join(output_dir, f'{title}_intermediate.tif.aux.xml')
                ]
                for file_path in files_to_delete:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    else:
                        print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error during cleanup: {e}")
            
            end_time = time.time()
            duration = end_time - start_time
            minutes, seconds = divmod(duration, 60)
            print(f"***** DONE! Total processing time: {int(minutes)} minutes and {seconds:.1f} seconds")
        except Exception as e:
            print(f"An error occurred while processing {csb}: {e}")

# Tkinter GUI setup
root = tk.Tk()
root.title("CSB Processing")

title_var = tk.StringVar()
csb_var = tk.StringVar()
BAG_filepath_var = tk.StringVar()
fp_zones_var = tk.StringVar()
output_dir_var = tk.StringVar()

tk.Label(root, text='2. Directory with Raw CSB data in *.csv format').grid(row=1, column=0, sticky='w')
csb_dir_entry = tk.Entry(root, textvariable=csb_var)
csb_dir_entry.grid(row=1, column=1)
tk.Button(root, text='Browse', command=lambda: open_folder_dialog(csb_var)).grid(row=1, column=2)

tk.Label(root, text='3b. Input BAG or GeoTiff file for comparison bathymetry').grid(row=3, column=0, sticky='w')
BAG_filepath_entry = tk.Entry(root, textvariable=BAG_filepath_var)
BAG_filepath_entry.grid(row=3, column=1)
tk.Button(root, text='Browse', command=lambda: open_file_dialog(BAG_filepath_var, [("BAG or GeoTIFF file", "*.bag;*.tif;*.tiff")])).grid(row=3, column=2)

tk.Label(root, text='4. Tide Zone file in *.shp format').grid(row=4, column=0, sticky='w')
fp_zones_entry = tk.Entry(root, textvariable=fp_zones_var)
fp_zones_entry.grid(row=4, column=1)
tk.Button(root, text='Browse', command=lambda: open_file_dialog(fp_zones_var, [("Shapefile", "*.shp")])).grid(row=4, column=2)

tk.Label(root, text='5. Specify output folder').grid(row=5, column=0, sticky='w')
output_dir_entry = tk.Entry(root, textvariable=output_dir_var)
output_dir_entry.grid(row=5, column=1)
tk.Button(root, text='Browse', command=lambda: open_folder_dialog(output_dir_var)).grid(row=5, column=2)

tk.Button(root, text='Process', command=process_csb_threaded).grid(row=6, column=1, sticky='e')

bluetopo_var = tk.BooleanVar()
bluetopo_checkbox = tk.Checkbutton(root, text="3a. Use Automated BlueTopo Download", variable=bluetopo_var)
bluetopo_checkbox.grid(row=2, column=0, columnspan=2, sticky='w')

def on_bluetopo_check():
    if bluetopo_var.get():
        BAG_filepath_entry.config(state='disabled')
    else:
        BAG_filepath_entry.config(state='normal')

bluetopo_checkbox.config(command=on_bluetopo_check)

if __name__ == "__main__":
    root.mainloop()
