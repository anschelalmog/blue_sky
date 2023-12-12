from modules.base_classes import BaseTraj
from .utils import get_mpd, cosd, sind
import numpy as np
import scipy.io as sp
import os
import matplotlib.pyplot as plt


# from pyulog import ULog


def generate_map(_map, flat=False):
    flat = True


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

        height_map = height_map + gaussian
        ################
        terrain_normalized = (terrain_tile - terrain_tile.min()) / (terrain_tile.max() - terrain_tile.min())

        height_map = (terrain_normalized * 1734).astype(int)

    # Plot
    plt.figure()
    plt.imshow(height_map, cmap='terrain')
    plt.title('Generated Height Map')
    plt.tight_layout()
    plt.show()

    return height_map


class LoadMap:
    def __init__(self, args):
        self.lat_bounds, self.lon_bounds, self.pos_final_lat, self.pos_final_lon = self.get_map_bounds(args)
        self.Lat, self.Lon, self.North, self.East, self.mpd_N, self.mpd_E = self.get_map_axis(args)
        self.map_grid = self.get_map(args)

    @staticmethod
    def get_map_bounds(args):
        north_to_lat, east_to_lon = get_mpd(args.lat)

        pos_final_lat = args.lat + args.avg_spd / north_to_lat * cosd(args.psi) * args.time_end
        pos_final_lon = args.lon + args.avg_spd / east_to_lon * sind(args.psi) * args.time_end

        init_lat = np.floor(np.min([args.lat, pos_final_lat]))
        init_lon = np.floor(np.min([args.lon, pos_final_lon]))
        final_lat = np.ceil(np.max([args.lat, pos_final_lat]))
        final_lon = np.ceil(np.max([args.lon, pos_final_lon]))

        return [init_lat, final_lat], [init_lon, final_lon], pos_final_lat, pos_final_lon

    def get_map_axis(self, args):
        init_lat, final_lat = self.lat_bounds[0], self.lat_bounds[1]
        init_lon, final_lon = self.lon_bounds[0], self.lon_bounds[1]

        rate = args.map_res / 3600

        map_lat = np.arange(init_lat, final_lat, rate)
        map_lat = np.append(map_lat, map_lat[-1] + rate)
        map_lon = np.arange(init_lon, final_lon, rate)
        map_lon = np.append(map_lon, map_lon[-1] + rate)

        north_to_lat, east_to_lon = get_mpd(map_lat)

        map_north = map_lat * north_to_lat
        map_east = map_lon * east_to_lon

        return map_lat, map_lon, map_north, map_east, north_to_lat, east_to_lon

    def get_map(self, args):
        # Load map data from specified tiles.
        min_lat_int, max_lat_int = int(np.floor(self.lat_bounds[0])), int(np.ceil(self.lat_bounds[1]))
        min_lon_int, max_lon_int = int(np.floor(self.lon_bounds[0])), int(np.ceil(self.lon_bounds[1]))

        # Determine tile size and format based on map resolution.
        tile_length, map_level, ext = (1200, 1, 'dt1') if args.map_res == 3 else (3600, 3, 'dt2')

        # load map
        n_north = max_lat_int - min_lat_int
        n_east = max_lon_int - min_lon_int
        map_full_tiles = np.zeros([n_north * tile_length + 1, n_east * tile_length + 1])
        for e in np.arange(min_lon_int, max_lon_int):
            for n in np.arange(min_lat_int, max_lat_int):
                tile_path = os.path.join(args.map_path, f'Level{map_level}', f'E0{e}', f'n{n}.mat')
                try:
                    tile_load = sp.loadmat(tile_path).get('data', np.zeros((tile_length + 1, tile_length + 1)))
                except FileNotFoundError:
                    tile_load = None
                    print(f'file not found: {tile_path}')

                x_idx = slice((n - min_lat_int) * tile_length, (n - min_lat_int + 1) * tile_length + 1)
                y_idx = slice((e - min_lon_int) * tile_length, (e - min_lon_int + 1) * tile_length + 1)

                map_full_tiles[x_idx, y_idx] = tile_load

                if np.all(map_full_tiles == 0) or np.all(np.isnan(map_full_tiles)):
                    map_full_tiles = generate_map(map_full_tiles)

        return map_full_tiles


class TrajFromFile(BaseTraj):
    def __init__(self, args):
        super().__init__(args)
        self.read_logs()

    def read_logs(self):
        pass
