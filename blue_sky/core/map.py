import os
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
from ..utils.math import cosd, sind, get_mpd
from ..utils.mockin_map import mocking_map

class Map:
    def __init__(self):
        self.meta: dict = None
        self.axis: dict = None
        self.bounds: dict = None
        self.mpd: dict = None
        self.grid = None
        self.mock: bool = False

    def __str__(self):
        if self.meta is None:
            return 'Map Class instance: not initialized'
        else:
            map_level = self.meta['map_level']
            return f'Map class instance: map level {map_level}, {self.bounds}'

    def load(self, args, mock=False):
        self.mock = mock
        required_attributes = ['maps_dir', 'map_res', 'results_folder', 'init_lat', 'init_lon', 'avg_spd', 'psi',
                               'time_end', 'acc_north', 'acc_east']
        for attr in required_attributes:
            if not hasattr(args, attr):
                raise AttributeError(f"Missing required attribute '{attr}' in 'args'.")

        tile_length, map_level, ext = (1200, 1, 'dt1') if args.map_res == 3 else (3600, 3, 'dt2')
        ext = 'mat'  # for run examples purpose
        self.meta = {
            'maps_dir': args.maps_dir,
            'map_res': args.map_res,
            'map_level': map_level,
            'rate': args.map_res / 3600,
            'tile_length': tile_length,
            'ext': ext,
            'out_folder': args.results_folder
        }
        self._set_map_boundaries(args)
        self._set_axis()
        self._create_grid()
        return self

    def _set_axis(self):
        """
        Sets the axis values for latitude and longitude based on the given bounds and metadata.

        :return: None
        """
        self.axis = {}

        map_lat = np.arange(self.bounds['lat'][0], self.bounds['lat'][1], self.meta['rate'])
        self.axis['lat'] = np.append(map_lat, map_lat[-1] + self.meta['rate'])

        map_lon = np.arange(self.bounds['lon'][0], self.bounds['lon'][1], self.meta['rate'])
        self.axis['lon'] = np.append(map_lon, map_lon[-1] + self.meta['rate'])

        self.mpd = {'north': (get_mpd(self.axis['lat']))[0], 'east': (get_mpd(self.axis['lat']))[1]}

        self.axis['north'] = self.axis['lat'] * self.mpd['north']
        self.axis['east'] = self.axis['lon'] * self.mpd['east']

    def _set_map_boundaries(self, args):
        """
        give a max estimation based on the vehicle velocity's what's the boundaries of the map would be
        :param args
        :return:  updates self. bounds based on the given arguments
        """
        mpd_N, mpd_E = get_mpd(args.init_lat)
        # X = X_0 + V_0 * t + 1/2 V*t^2
        pos_final_lat = (args.init_lat + (((args.avg_spd * sind(args.psi)) * args.time_end) +
                                          (0.5 * args.acc_north * args.time_end ** 2)) / mpd_N)

        pos_final_lon = (args.init_lon + (((args.avg_spd * cosd(args.psi)) * args.time_end) +
                                          (0.5 * args.acc_east * args.time_end ** 2)) / mpd_E)

        init_lat = np.floor(np.min([args.init_lat, pos_final_lat]))
        init_lat = init_lat.astype(int)
        final_lat = np.ceil(np.max([args.init_lat, pos_final_lat]))
        final_lat = final_lat.astype(int)
        init_lon = np.floor(np.min([args.init_lon, pos_final_lon]))
        init_lon = init_lon.astype(int)
        final_lon = np.ceil(np.max([args.init_lon, pos_final_lon]))
        final_lon = final_lon.astype(int)

        self.bounds = {'lat': [init_lat, final_lat], 'lon': [init_lon, final_lon]}
        # self.final_pos = [pos_final_lat, pos_final_lon]

    def _create_grid(self, lat=None, lon=None):

        # creating a new grid in running time
        if lat is not None and lon is not None:
            min_lat, max_lat = np.floor(lat), np.ceil(lat)
            min_lon, max_lon = np.floor(lon), np.ceil(lon)

        # creating a new grind before running time
        else:
            min_lat, max_lat = min(self.bounds['lat']), max(self.bounds['lat'])
            min_lon, max_lon = min(self.bounds['lon']), max(self.bounds['lon'])

        map_full_tiles = np.zeros(
            [abs(max_lat - min_lat) * self.meta['tile_length'] + 1,
             abs(max_lon - min_lon) * self.meta['tile_length'] + 1])

        # Load map tiles and assemble the full map
        lat_range, lon_range = range(min_lat, max_lat), range(min_lon, max_lon)

        for e in lon_range:
            for n in lat_range:
                tile_path = os.path.join(os.getcwd(), self.meta['maps_dir'],
                                         f'Level{self.meta["map_level"]}', 'DTED',
                                         f'E0{e}', f'n{n}.{self.meta["ext"]}')
                tile_load = self._load_tile(tile_path, self.meta['tile_length'], e, n)

                # Define indices for placing the tile in the full map
                x_idx = slice((n - min_lat) * self.meta['tile_length'], (n - min_lat + 1) *
                              self.meta['tile_length'] + 1)
                y_idx = slice((e - min_lon) * self.meta['tile_length'], (e - min_lon + 1) *
                              self.meta['tile_length'] + 1)

                map_full_tiles[x_idx, y_idx] = tile_load

        self._validate_map(map_full_tiles)

    @staticmethod
    def _load_tile(tile_path, tile_length, e, n):
        try:
            ret = sp.loadmat(tile_path)['elevation_data']
            print(f'Loaded tile: E{e}N{n}')
            return ret
        except FileNotFoundError:
            print(f'file not found: {tile_path}')
            return None

    def _validate_map(self, map_full_tiles):
        if np.all(map_full_tiles == 0) or np.all(np.isnan(map_full_tiles)) or self.mock:
            self.grid = mocking_map(map_full_tiles)
        else:
            self.grid = map_full_tiles.astype(int)

    def update_map(self, new_lat, new_lon):
        # Error Handling for invalid inputs
        if not isinstance(new_lat, (int, float)) or not isinstance(new_lon, (int, float)):
            raise ValueError("Invalid latitude or longitude. Both should be numeric.")

        new_lat_start, new_lat_end = np.floor(new_lat), np.ceil(new_lat)
        new_lon_start, new_lon_end = np.floor(new_lon), np.ceil(new_lon)

        # Determine if an update is necessary
        boundary_updated = False
        if new_lat_start < self.bounds['lat'][0] or new_lat_end > self.bounds['lat'][1]:
            init_lat = min(self.bounds['lat'][0], new_lat_start)
            final_lat = max(self.bounds['lat'][1], new_lat_end)
            self.bounds['lat'] = [init_lat, final_lat]
            boundary_updated = True

        if new_lon_start < self.bounds['lon'][0] or new_lon_end > self.bounds['lon'][1]:
            init_lon = min(self.bounds['lon'][0], new_lon_start)
            final_lon = max(self.bounds['lon'][1], new_lon_end)
            self.bounds['lon'] = [init_lon, final_lon]
            boundary_updated = True

        # Regenerate the grid and axis only if the boundaries were updated
        if boundary_updated:
            self._set_axis()
            self._create_grid()
            print(f"Map boundaries updated to lat: {self.bounds['lat']}, lon: {self.bounds['lon']}")
            print("Grid and axis regenerated.")
        else:
            print("New coordinates within existing boundaries. No update needed.")

    def _update_map_attributes(self, new_map, axis):
        """
            not in use anymore, concatenate the new map axis and grid to the old one

        :param new_map: instance of Map class with no over-laping attributes to the old ones
        :param axis: the axis it should be concatenated to
        :return:
        """
        if axis == 'north':
            self.grid = np.concatenate((self.grid, new_map.grid), axis=0)
            self.axis['north'] = np.concatenate(self.bounds['north'], new_map.bounds['north'])
            self.axis['lat'] = np.concatenate(self.bounds['lat'], new_map.bounds['lat'])
            self.mpd['north'] = np.concatenate(self.mpd['north'], new_map.mpd['north'])
            self.bounds['lat'][1] = np.ceil(max(self.axis['lat']))

        elif axis == 'south':
            self.grid = np.concatenate((new_map.grid, self.grid), axis=0)
            self.axis['north'] = np.concatenate(new_map.bounds['north'], self.bounds['north'])
            self.axis['lat'] = np.concatenate(new_map.bounds['lat'], self.bounds['lat'])
            self.mpd['north'] = np.concatenate(new_map.mpd['north'], self.mpd['north'])
            self.bounds['lat'][0] = np.floor(min(self.axis['lat']))

        elif axis == 'east':
            self.grid = np.concatenate((new_map.grid, self.grid), axis=1)
            self.axis['east'] = np.concatenate(new_map.bounds['east'], self.bounds['east'])
            self.axis['lon'] = np.concatenate(new_map.bounds['lon'], self.bounds['lon'])
            self.mpd['east'] = np.concatenate(new_map.mpd['east'], self.mpd['east'])
            self.bounds['lon'][1] = np.ceil(max(self.axis['lon']))

        else:  # axis == 'west':
            self.grid = np.concatenate((new_map.grid, self.grid), axis=1)
            self.axis['east'] = np.concatenate(new_map.bounds['east'], self.bounds['east'])
            self.axis['lon'] = np.concatenate(self.bounds['lon'], new_map.bounds['lon'])
            self.mpd['east'] = np.concatenate(self.mpd['east'], new_map.mpd['east'])
            self.bounds['lon'][0] = np.floor(min(self.axis['lon']))

    def save(self, file_path):
        """
        Saves the current Map instance to a .mat file.

        :param file_path: The path (including file name) where the .mat file will be saved.
        """
        data_to_save = {
            'grid': self.grid,
            'meta': self.meta,
            'axis': self.axis,
            'bounds': self.bounds,
            'mpd': self.mpd
        }

        try:
            sp.savemat(file_path, data_to_save)
            print(f"Map saved successfully to {file_path}")
        except IOError as e:
            print(f"Failed to save the map due to an I/O error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving the map: {e}")

    def visualize_map(self, mode, save=False):
        """
        Visualizes the map in either 2D or 3D mode.

        :param mode: A string that determines the visualization mode.
                     '2D' for a two-dimensional plot and '3D' for a three-dimensional plot.
        """
        if self.grid is None:
            print("Map grid is not initialized.")
            return

        if mode == '2D':  # 2D visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(self.grid, extent=(self.axis['east'].min(), self.axis['east'].max(),
                                          self.axis['north'].min(), self.axis['north'].max()), origin='lower')
            title = 'Map Visualization (2D)'
            plt.title(title)
            plt.xlabel('East')
            plt.ylabel('North')
            plt.colorbar(label='Elevation')
            plt.grid(True)

            if save:
                plt.savefig(os.path.join(self.meta['out_folder'], f'{title}.svg'))
                plt.savefig(os.path.join(self.meta['out_folder'], f'{title}.png'))

            plt.show()

        elif mode == '3D':  # 3D visualization
            North, East = np.meshgrid(self.axis['north'], self.axis['east'])

            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(East, North, self.grid.T, cmap='terrain', edgecolor='none')
            title = 'Map Visualization (3D)'
            ax.set_title(title)
            ax.set_xlabel('East')
            ax.set_ylabel('North')
            ax.set_zlabel('Elevation')

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            if save:
                plt.savefig(os.path.join(self.meta['out_folder'], f'{title}.svg'))
                plt.savefig(os.path.join(self.meta['out_folder'], f'{title}.png'))

            plt.show()
        else:
            print("Invalid mode selected. Please choose '2D' or '3D'.")


