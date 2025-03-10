# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:18:55 2024

@author: Anthony.R.Klemm
"""

import duckdb

con = duckdb.connect(r'E:\csb\kuskokwim\processed\csb_kuskokwim.duckdb')


# Drop the incorrect 'depth_mod' column if it exists
columns_query = "DESCRIBE csb"
columns_df = con.execute(columns_query).fetchdf()
if 'depth_mod' in columns_df['column_name'].values:
    print("Dropping incorrect 'depth_mod' column in DuckDB...")
    con.execute("ALTER TABLE csb DROP COLUMN depth_mod")
if 'uncertainty_vert' in columns_df['column_name'].values:
    print("Dropping old 'uncertainty_vert' column in DuckDB...")
    con.execute("ALTER TABLE csb DROP COLUMN uncertainty_vert")
if 'uncertainty_hori' in columns_df['column_name'].values:
    print("Dropping old 'uncertainty_hori' column in DuckDB...")
    con.execute("ALTER TABLE csb DROP COLUMN uncertainty_hori")
    
# Add a new column 'depth_mod' to the table
con.execute("ALTER TABLE csb ADD COLUMN depth_mod DOUBLE;")

# Update the 'depth_mod' column with calculated values
con.execute("""
UPDATE csb
SET depth_mod = (depth_new - sub.average_diff) * -1
FROM (
    SELECT unique_id, AVG(diff) AS average_diff
    FROM csb
    WHERE diff > -12 AND diff < 12 AND Raster_Value > -20 AND Uncertainty_Value < 3
    GROUP BY unique_id
) AS sub
WHERE csb.unique_id = sub.unique_id;
""")

# SQL command to update 'depth_mod' where it equals 0
update_query = """
UPDATE csb
SET depth_mod = depthfinal
WHERE depth_mod = 0;
"""

# Execute the fill command
con.execute(update_query)
print("Updated depth_mod values where they were zero with depthfinal.")


uncert_vert_query = """
UPDATE csb
SET uncertainty_vert = (2 + (depth_mod * -0.05))
"""

uncert_hori_query="""
UPDATE csb
SET uncertainty_hori = 10
"""

# Add new column 'uncertainty_vert' to the csb table
con.execute("ALTER TABLE csb ADD COLUMN uncertainty_vert DOUBLE;")

# Add new column 'uncertainty_hori' to the csb table
con.execute("ALTER TABLE csb ADD COLUMN uncertainty_hori DOUBLE;")

# Execute uncertainty calculation command
con.execute(uncert_vert_query)
con.execute(uncert_hori_query)
print("Uncertainty values calculated to CATZOC C")



con.close()
