import os
import numpy as np
import rasterio
from scipy.io import savemat
import matplotlib.pyplot as plt

with rasterio.open('C:/Users/ALMOGAN/Downloads/gt30e020n40_dem/gt30e020n40.dem') as dem:
    elevation_data = dem.read(1)
show_details = True

if show_details:
    print(elevation_data.shape)
    print(elevation_data.dtype)
    print(elevation_data)
    plt.imshow(elevation_data, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Elevation Map')
    plt.show()

break_and_save = True
if break_and_save:
    elevation_data[elevation_data == -9999] = 0
    elevation_data[elevation_data < 0] = np.abs(elevation_data[elevation_data < 0])

    # Define the size of the output sections
    section_size = 1201
    # Calculate the number of sections to create
    num_sections_x = elevation_data.shape[1] // (section_size - 1) - 1
    num_sections_y = elevation_data.shape[0] // (section_size - 1) - 1

    # Create the directories and save the .mat files
    base_path = 'C:/Users/ALMOGAN/Documents/blue_sky/Map/Level1/DTED/'

    for i in range(num_sections_y):
        for j in range(num_sections_x):
            # Define the section boundaries, including overlap
            start_x = j * (section_size - 1)
            end_x = start_x + section_size
            start_y = i * (section_size - 1)
            end_y = start_y + section_size
            # Extract the section
            section = elevation_data[start_y:end_y, start_x:end_x]

            # Calculate the latitude and longitude for the section
            lat = 40 - i  # Assuming each section corresponds to a 1 degree change in latitude
            lon = 20 + j  # Assuming each section corresponds to a 1 degree change in longitude

            # Create the directory name based on the longitude
            dir_name = f'E{lon:03d}'
            dir_path = os.path.join(base_path, dir_name)
            # Make the directory if it doesn't exist
            os.makedirs(dir_path, exist_ok=True)

            # Define the filename based on the latitude
            file_name = f'n{lat:02d}.mat'
            file_path = os.path.join(dir_path, file_name)

            # Save the section as a .mat file
            savemat(file_path, {'elevation_data': section})
