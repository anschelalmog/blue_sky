import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import os

from modules.utils import get_mpd, cosd, sind
from modules.base_traj import BaseTraj


def generate_map(_map, flat=True, to_plot=True):
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


class Map:
    def __init__(self, args):
        self.args = args
        self.lat_bounds = None
        self.lon_bounds = None
        self.final_pos = None
        self.ax_lat = None
        self.ax_lon = None
        self.ax_north = None
        self.ax_east = None
        self.mpd_north = None
        self.mpd_east = None
        self.grid = None

    def load(self):
        self._set_map_boundaries()
        self._set_axis()
        self._create_grid()
        return self

    def _set_map_boundaries(self):
        mpd_N, mpd_E = get_mpd(self.args.init_lat)

        pos_final_lat = self.args.init_lat + self.args.avg_spd / mpd_N * sind(self.args.psi) * self.args.time_end
        pos_final_lon = self.args.init_lon + self.args.avg_spd / mpd_E * cosd(self.args.psi) * self.args.time_end

        init_lat = np.floor(np.min([self.args.init_lat, pos_final_lat]))
        final_lat = np.ceil(np.max([self.args.init_lat, pos_final_lat]))
        init_lon = np.floor(np.min([self.args.init_lon, pos_final_lon]))
        final_lon = np.ceil(np.max([self.args.init_lon, pos_final_lon]))

        self.lat_bounds = [init_lat, final_lat]
        self.lon_bounds = [init_lon, final_lon]
        self.final_pos = [pos_final_lat, pos_final_lon]

    def _set_axis(self):
        rate = self.args.map_res / 3600
        init_lat, final_lat = self.lat_bounds[0], self.lat_bounds[1]
        init_lon, final_lon = self.lon_bounds[0], self.lon_bounds[1]

        map_lat = np.arange(init_lat, final_lat, rate)
        self.ax_lat = np.append(map_lat, map_lat[-1] + rate)
        map_lon = np.arange(init_lon, final_lon, rate)
        self.ax_lon = np.append(map_lon, map_lon[-1] + rate)

        self.mpd_north, self.mpd_east = get_mpd(self.ax_lat)

        self.ax_north = self.ax_lat * self.mpd_north
        self.ax_east = self.ax_lon * self.mpd_east

    def _create_grid(self):
        # Initialize bounds and tile settings
        min_lat_int, max_lat_int = map(int, (np.floor(self.lat_bounds[0]), np.ceil(self.lat_bounds[1])))
        min_lon_int, max_lon_int = map(int, (np.floor(self.lon_bounds[0]), np.ceil(self.lon_bounds[1])))
        tile_length, map_level, ext = (1200, 1, 'dt1') if self.args.map_res == 3 else (3600, 3, 'dt2')
        ext = 'mat'
        # Create an empty map grid
        map_full_tiles = np.zeros(
            [(max_lat_int - min_lat_int) * tile_length + 1, (max_lon_int - min_lon_int) * tile_length + 1])

        # Load map tiles and assemble the full map
        for e in range(min_lon_int, max_lon_int):
            for n in range(min_lat_int, max_lat_int):
                tile_path = os.path.join(os.getcwd(), self.args.maps_dir, f'Level{map_level}', 'DTED', f'E0{e}',
                                         f'n{n}.{ext}')
                tile_load = self._load_tile(tile_path, tile_length)

                # Define indices for placing the tile in the full map
                x_idx = slice((n - min_lat_int) * tile_length, (n - min_lat_int + 1) * tile_length + 1)
                y_idx = slice((e - min_lon_int) * tile_length, (e - min_lon_int + 1) * tile_length + 1)

                map_full_tiles[x_idx, y_idx] = tile_load

        self._validate_map(map_full_tiles)

    @staticmethod
    def _load_tile(tile_path, tile_length):
        try:
            return sp.loadmat(tile_path).get('data', np.zeros((tile_length + 1, tile_length + 1)))
        except FileNotFoundError:
            print(f'file not found: {tile_path}')
            return None

    def _validate_map(self, map_full_tiles):
        if np.all(map_full_tiles == 0) or np.all(np.isnan(map_full_tiles)):
            self.grid = generate_map(map_full_tiles)
        else:
            self.grid = map_full_tiles


class TrajFromFile(BaseTraj):
    pass
