CSB data pipeline for scraping, tide correction, spatial database population (duckdb-spatial), vertical offset analysis/correction, uncertainty estimation, vessel speed estimation and PMM imputation-based outlier detection algorithms, and exporting data as vessel-transit geopackages and geotiff DEMs. 

There are other helper scripts, and some scripts for dashboard/leaderboard creation as well. 

You're going to need a shapefile of the CO-OPS discrete zoned tide model. One is provided in the OCS Pydro distribution, along with the main CSB processing script. 

Preferred method is to use the BlueTopo bathymetry as the reference bathy, which will be downloaded automatically based on the input raw data coverage, but it also allows the user to use a BAG instead (I think only SR BAGs are supported at this time, but VR BAGs will be supported soon). 
