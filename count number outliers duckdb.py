# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:19:51 2025

@author: Anthony.R.Klemm
"""

import duckdb
# Path to your DuckDB database
duckdb_path = r'D:/CSB_texas/processed/csb_texas.duckdb'

# Connect to DuckDB
conn = duckdb.connect(duckdb_path)

try:
    # Query to count rows where Outlier is TRUE
    result = conn.execute("SELECT COUNT(*) AS outlier_count FROM csb WHERE Outlier = 1").fetchone()
    print(f"Count of records where Outlier = TRUE: {result[0]}")

    # query to count rows where outlier is false
    result = conn.execute("SELECT COUNT(*) AS outlier_count FROM csb WHERE Outlier = FALSE").fetchone()
    print(f"Count of records where Outlier = FALSE: {result[0]}")

finally:
    # Close the connection
    conn.close()
