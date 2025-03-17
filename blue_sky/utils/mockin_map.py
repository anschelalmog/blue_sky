import numpy as np
import matplotlib.pyplot as plt

def mocking_map(_map, flat=True, to_plot=False):
    tile_size = _map.shape[0]
    if flat:
        height_map = np.full((tile_size, tile_size), 1000)
    else:
        min_height = 0
        max_height = 1203

        x = np.linspace(0, 1, tile_size)
        y = np.linspace(0, 1, tile_size)
        x_ax, y_ax = np.meshgrid(x, y)

        frequency_y = 3
        frequency_x = 5
        amplitude = 180
        terrain_tile = amplitude * np.sin(2 * np.pi * frequency_x * x_ax)
        terrain_tile += amplitude * np.sin(2 * np.pi * frequency_y * y_ax)

        terrain_tile = np.clip(terrain_tile, min_height, max_height)
        noise = np.random.uniform(-30, 250, terrain_tile.shape)
        terrain_tile += noise
        terrain_normalized = (terrain_tile - terrain_tile.min()) / (terrain_tile.max() - terrain_tile.min())
        height_map = (terrain_normalized * 1023).astype(int)

        #####################
        std_dev = 350
        amplitude = 700

        rows, cols = height_map.shape
        center_row, center_col = rows // 2, cols // 2
        y, x = np.indices((rows, cols))
        gaussian = amplitude * np.exp(-((x - center_col) ** 2 + (y - center_row) ** 2) / (2 * std_dev ** 2))

        terrain_tile = height_map + gaussian
        ################
        terrain_normalized = (terrain_tile - terrain_tile.min()) / (terrain_tile.max() - terrain_tile.min())

        height_map = (terrain_normalized * 1734).astype(int)

    if to_plot:
        plt.figure()
        plt.imshow(height_map, cmap='terrain')
        plt.title('Generated Height Map')
        plt.tight_layout()
        plt.show()

    return height_map

