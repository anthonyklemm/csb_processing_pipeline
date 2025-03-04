# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:38:58 2024

@author: Anthony.R.Klemm
"""
import os
import duckdb

# Path to your directory containing geopackages
directory_path = r"E:\csb\kuskokwim\processed"


# Connect to DuckDB - if it doesn't exist, it will create the *duckdb file for you
con = duckdb.connect(r"E:\csb\kuskokwim\processed\csb_kuskokwim.duckdb")
con.install_extension('spatial')
con.load_extension('spatial')

# Initialize a variable to check if the table has been created
table_created = False

# for filename in os.listdir(directory_path):
#     print(filename)

# Loop through each file in the directory
for filename in os.listdir(directory_path): 
    if filename.endswith('.gpkg'):
        geopackage_path = os.path.join(directory_path, filename)
        print(f'loading {geopackage_path} into duckdb table')
        if not table_created:
            # Create the table with the first geopackage
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS csb AS 
                SELECT * FROM ST_Read('{geopackage_path}')
            """)
            table_created = True
        else:
            # Insert subsequent geopackages into the existing table
            con.execute(f"""
                INSERT INTO csb
                SELECT * FROM ST_Read('{geopackage_path}')
            """)
        print(f'{geopackage_path} loaded successfully')
# Optionally, check the data
print(con.table('csb'))

