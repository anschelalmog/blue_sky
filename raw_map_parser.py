import os
import numpy as np
import rasterio
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

laptop_run = False
show_details = False
break_save_dt1 = False
break_save_dt2 = True
compare_res = True

if laptop_run:
    dem_file_path = 'C:/Users/ALMOGAN/Downloads/gt30e020n40_dem/gt30e020n40.dem'
    dt1_out_path = 'C:/Users/ALMOGAN/Documents/blue_sky/Map/Level1/DTED/'
    dt2_out_path = 'C:/Users/ALMOGAN/Documents/blue_sky/Map/Level2/DTED/'
else:
    dem_file_path = 'D:/python_project/blue_sky/Map/source_dem/gt30e020n40_dem/gt30e020n40.dem'
    dt1_out_path = 'D:/python_project/blue_sky/Map/Level1/DTED/'
    dt2_out_path = 'D:/python_project/blue_sky/Map/Level2/DTED/'

with rasterio.open(dem_file_path) as dem:
    elevation_data = dem.read(1)
    elevation_data[elevation_data == -9999] = 0
    elevation_data[elevation_data < 0] = np.abs(elevation_data[elevation_data < 0])

if show_details:
    print(elevation_data.shape)
    print(elevation_data.dtype)
    print(elevation_data)
    plt.imshow(elevation_data, cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Elevation Map')
    plt.show()

if break_save_dt1:
    section_size = 1201
    num_sections_x = elevation_data.shape[1] // (section_size - 1) - 1
    num_sections_y = elevation_data.shape[0] // (section_size - 1) - 1

    base_path = dt1_out_path

    for i in range(num_sections_y):
        for j in range(num_sections_x):
            start_x = j * (section_size - 1)
            end_x = start_x + section_size
            start_y = i * (section_size - 1)
            end_y = start_y + section_size
            # Extract the section
            section = elevation_data[start_y:end_y, start_x:end_x]

            # Calculate the latitude and longitude for the section
            lat = 40 - i
            lon = 20 + j

            dir_name = f'E{lon:03d}'
            dir_path = os.path.join(base_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

            # Define the filename based on the latitude
            file_name = f'n{lat:02d}.mat'
            file_path = os.path.join(dir_path, file_name)

            savemat(file_path, {'elevation_data': section})

if break_save_dt2:
    section_size = 3601
    num_sections_x = elevation_data.shape[1] // (section_size - 1) - 1
    num_sections_y = elevation_data.shape[0] // (section_size - 1) - 1

    base_path = dt2_out_path

    for i in range(num_sections_y):
        for j in range(num_sections_x):
            start_x = j * (section_size - 1)
            end_x = start_x + section_size
            start_y = i * (section_size - 1)
            end_y = start_y + section_size
            # Extract the section
            section = elevation_data[start_y:end_y, start_x:end_x]

            # Calculate the latitude and longitude for the section
            lat = 40 - i
            lon = 20 + j

            dir_name = f'E{lon:03d}'
            dir_path = os.path.join(base_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)

            # Define the filename based on the latitude
            file_name = f'n{lat:02d}.mat'
            file_path = os.path.join(dir_path, file_name)

            savemat(file_path, {'elevation_data': section})

if compare_res:

    dt1_file_path = os.path.join(dt1_out_path, f'E{22:03d}', f'n{38:02d}.mat')
    dt2_file_path = os.path.join(dt2_out_path, f'E{22:03d}', f'n{38:02d}.mat')

    if os.path.exists(dt1_file_path) and os.path.exists(dt2_file_path):
        dt1_data = loadmat(dt1_file_path)['elevation_data']
        dt2_data = loadmat(dt2_file_path)['elevation_data']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(dt1_data, cmap='terrain')
        ax1.set_title('DT1 Tile')
        ax1.axis('off')
        ax2.imshow(dt2_data, cmap='terrain')
        ax2.set_title('DT2 Tile')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"DT1 or DT2 tile file not found: {dt1_file_path}, {dt2_file_path}")