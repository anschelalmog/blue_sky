"""
This script processes a DEM file and its associated header file to generate DT1 or DT2 files.
DT1 files are 1201x1201 matrices where each element represents a 100m x 100m area.
DT2 files are 3601x3601 matrices where each element represents a 33m x 33m area.
The script performs the following steps:
1. Reads the DEM and .hdr files.
2. Extracts necessary metadata from the .hdr file.
3. Calculates the east and north axes based on the metadata.
4. Iterates over the DEM data in chunks corresponding to the specified tile size (DT1 or DT2).
5. Resamples each chunk to match the desired resolution using interpolation.
6. Saves the processed tiles to .mat files in a directory structure based on the level_map value:
   - level_map = 3: DT1 files are saved to 'Map/Level1/DTED/E024/n13.mat'.
   - level_map = 2: DT2 files are saved to 'Map/Level2/DTED/E024/n13.mat'.
"""

import os
import numpy as np
import rasterio
from scipy.ndimage import zoom
from scipy.io import savemat

## inputs
dem_file_path = '.dem'
hdr_file_path = '.hdr'
level_map = 3  # Change this value to 2 for DT2 specifications

degree_to_meters = 111000

# Reading the .hdr file
with open(hdr_file_path, 'r') as hdr_file:
    hdr_content = hdr_file.readlines()

# Extracting necessary information from the .hdr file
hdr_info = {}
for line in hdr_content:
    key, value = line.split()
    hdr_info[key] = float(value)

# Reading the .dem file using rasterio
with rasterio.open(dem_file_path) as dataset:
    dem_array = dataset.read(1)

# Extracting the required fields from the hdr_info
nrows = int(hdr_info['NROWS'])
ncols = int(hdr_info['NCOLS'])
ulxmap = hdr_info['ULXMAP']
ulymap = hdr_info['ULYMAP']
xdim = hdr_info['XDIM']
ydim = hdr_info['YDIM']

# Calculate the east and north axes
east_axis = ulxmap + np.arange(ncols) * xdim
north_axis = ulymap - np.arange(nrows) * ydim

# Specifications based on level_map
if level_map == 3:
    tile_size = 1201
    resolution_m = 100
    base_dir = 'Map/Level1/DTED'
elif level_map == 2:
    tile_size = 3601
    resolution_m = 33
    base_dir = 'Map/Level2/DTED'
else:
    raise ValueError("Invalid level_map value. Use 3 for DT1 and 2 for DT2.")

# DEM file resolution in meters
dem_x_resolution_m = hdr_info['XDIM'] * degree_to_meters
dem_y_resolution_m = hdr_info['YDIM'] * degree_to_meters

# Calculate the ratio
ratio_x = dem_x_resolution_m / resolution_m
ratio_y = dem_x_resolution_m / resolution_m

# Create directories if they do not exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


def process_and_save_tile(tile_data, east_index, north_index):
    # Compute the shape of the output DT1/DT2 array
    target_shape = (tile_size, tile_size)

    # Resample the tile data using interpolation to match the target resolution
    resampled_array = zoom(tile_data, (target_shape[0] / tile_data.shape[0], target_shape[1] / tile_data.shape[1]),
                           order=1)

    # Save the data to a .mat file
    east_label = f"E{east_index:03d}"
    north_label = f"n{north_index:02d}"
    output_path = os.path.join(base_dir, east_label, north_label + '.mat')

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to .mat file
    savemat(output_path, {'data': resampled_array, 'east_axis': east_axis, 'north_axis': north_axis})

    print(f"Saved data and axes to {output_path}")


# Iterate over the DEM array in chunks corresponding to the DT1/DT2 tiles
for i in range(0, nrows, tile_size):
    for j in range(0, ncols, tile_size):
        # Extract the current tile
        tile_data = dem_array[i:i + tile_size, j:j + tile_size]

        # Process and save the tile
        east_index = int(ulxmap + j * xdim)
        north_index = int(ulymap - i * ydim)
        process_and_save_tile(tile_data, east_index, north_index)
