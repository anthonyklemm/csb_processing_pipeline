# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:47:26 2024

@author: Anthony.R.Klemm
"""

import geopandas as gpd
import requests
import time
import os


def ensure_grid_id_exists(gdf):
    if 'GRID_ID' not in gdf.columns:
        print("GRID_ID field does not exist. Creating and populating it with integers.")
        gdf['GRID_ID'] = ["csb_raw_" + str(i) for i in range(1, len(gdf) + 1)]
    else:
        print("GRID_ID field exists.")
    return gdf

def process_tile(bbox, email, tile_name, output_directory):
    print(f"Processing GRID_ID {tile_name} with bbox: {bbox}")
    payload = {
        "email": email,
        "bbox": bbox,
        "datasets": [
            {
                "label": "csb"
            }
        ]
    }

    # Submit the order
    response = requests.post('https://q81rej0j12.execute-api.us-east-1.amazonaws.com/order', json=payload)

    if response.status_code == 201:
        order_response = response.json()
        print("Debug - Order Response:", order_response)

        # Use the 'url' field directly from the response for status checking
        status_url = order_response.get('url', '')
        print(f"Using status URL: {status_url}")  # Debug print to verify the status URL

        # Introduce a delay before checking the status for the first time
        time.sleep(30)
    else:
        print(f"Failed to create order for GRID_ID {tile_name}:", response.text)
        return

    # Wait for order completion; check order status and download CSV
    retry_count = 0
    max_retries = 5
    status = 'initialized'
    
    while status.lower() not in ['complete', 'error'] and retry_count < max_retries:
        status, output_location = check_order_status(status_url)
        if status.lower() == 'complete':
            print(f"Order completed for GRID_ID {tile_name}. Download data from: {output_location}")
            download_url = 'https://order-pickup.s3.amazonaws.com/' + output_location[18:]
            local_file_path = os.path.join(output_directory, f'{tile_name}.csv')
            if download_csv(download_url, local_file_path):
                print(f"CSV file for GRID_ID {tile_name} processing can start now.")
            break
        elif status.lower() == 'error':
            print(f"Error in processing the order for GRID_ID {tile_name}.")
            break
        else:
            print(f"Order for GRID_ID {tile_name} is still processing. Waiting... (Attempt {retry_count + 1}/{max_retries})")
            time.sleep(15)
            retry_count += 1

    if retry_count == max_retries:
        print(f"Order for GRID_ID {tile_name} did not complete after {max_retries} attempts.")


def check_order_status(order_url):
    print("Checking status at:", order_url)  # Debug print

    for attempt in range(10):
        response = requests.get(order_url)
        print(f"Attempt {attempt + 1}, HTTP status code: {response.status_code}")  # Debug print

        if response.status_code == 200:
            try:
                status_response = response.json()
                status = status_response.get('status', 'error')
                output_location = status_response.get('output_location')
                print(f"Order status: {status}, Output location: {output_location}")
                return status, output_location
            except ValueError:
                print("Error parsing JSON response:", response.text)
                return 'error', None
        else:
            print(f"Failed to check order status. HTTP status code: {response.status_code}. Retrying in 15 seconds...")
            time.sleep(15)  # Wait 15 seconds before retrying

    # If all ten attempts fail, return an error status
    print("Failed to retrieve a valid order status after 10 attempts.")
    return 'error', None



# Download CSV from the provided URL
def download_csv(url, local_file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
        print(f"CSV file has been downloaded to {local_file_path}")
        return True
    else:
        print(f"Failed to download CSV. HTTP status code: {response.status_code}")
        return False


# Check if the CSV file already exists (this prevents tiles to be downloaded again if the program crashed halfway through)
def csv_file_exists(tile_name, output_directory):
    # Construct the path where the CSV file would be saved
    csv_file_path = os.path.join(output_directory, f"{tile_name}.csv")
    # Check if the file exists
    return os.path.exists(csv_file_path)


def main(input_polygon_path, output_directory, email):
    gdf = gpd.read_file(input_polygon_path).to_crs(epsg=4326)

    # Ensure GRID_ID exists and is populated
    gdf = ensure_grid_id_exists(gdf)

    # Iterate through each polygon in the GeoDataFrame
    for index, row in gdf.iterrows():
        polygon = row['geometry']
        tile_name = row['GRID_ID']
        
        # Check if the CSV file for the current tile already exists
        if csv_file_exists(tile_name, output_directory):
            print(f"CSV file for GRID_ID {tile_name} already exists. Skipping download.")
            continue  # Skip to the next iteration of the loop
        
        minx, miny, maxx, maxy = polygon.bounds
        bbox = f"{minx},{miny},{maxx},{maxy}"
        print(bbox)  # Debugging print for the bounding box

        # Now call process_tile for each tile
        process_tile(bbox, email, tile_name, output_directory)

if __name__ == "__main__":
    input_polygon_path = r"D:\csb\raw\Southeast\Southeast.shp"  # Replace with your input polygon path
    output_directory = r"D:\csb\raw\Southeast_22OCT2024" # Replace with your output directory
    email = 'anthony.r.klemm@noaa.gov'  # Replace with your email
    
    main(input_polygon_path, output_directory, email)
