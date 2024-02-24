import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sp
import os
import argparse

from src.utils import get_mpd, cosd, sind, mocking_map
from src.base_traj import BaseTraj


def set_settings():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--traj_from_file', type=bool, default=False,
                        help='load the trajectory from file or to generate one')
    parser.add_argument('--traj_path', type=str, default='', help='the path of the trajectory file')
    parser.add_argument('--plot_results', type=bool, default=True,
                        help='plot and save plots results at the end of the run')
    parser.add_argument('--noise_type', type=str, default='normal',
                        help='measurements noise type: none, normal or uniform')

    # Map Settings
    parser.add_argument('--maps_dir', type=str, default='Map',
                        help='Path to maps, format: "Map/LevelX/DTED/E0XX/mXX.mat".')
    parser.add_argument('--map_level', type=int, default=1, help='map level')

    # Time Settings
    parser.add_argument('--time_init', type=int, default=0, help='times starts counting, in [sec]')
    parser.add_argument('--time_end', type=int, default=100, help='times ends counting, in [sec]')
    # parser.add_argument('--time_rate', type=float, default=1, help='time rate, in [sec]')
    parser.add_argument('--time_res', type=float, default=0.1, help='flights resolution speed, in [sec]')

    # Errors Flags
    parser.add_argument('--flg_err_pos', type=bool, default=False, help='flag error for position')
    parser.add_argument('--flg_err_vel', type=bool, default=False, help='flag for error for velocity')
    parser.add_argument('--flg_err_alt', type=bool, default=False, help='flag error for altimeter')
    parser.add_argument('--flg_err_eul', type=bool, default=False, help='flag error for euler angels')
    parser.add_argument('--flg_err_baro_noise', type=bool, default=False, help='flag error for barometer noise')
    parser.add_argument('--flg_err_baro_bias', type=bool, default=False, help='flag error for barometer bias')

    # Errors Values
    parser.add_argument('--val_err_pos', type=float, default=200, help='error for position, in [m]')
    parser.add_argument('--val_err_vel', type=float, default=2, help='error for velocity, in [m/s]')
    parser.add_argument('--val_err_alt', type=float, default=5, help='error for altimeter, in [m]')
    parser.add_argument('--val_err_eul', type=float, default=0.05, help='error for euler angels, in [deg]')
    parser.add_argument('--val_err_baro_noise', type=float, default=5, help='error for barometer, in [m]')
    parser.add_argument('--val_err_baro_bias', type=float, default=5, help='error for barometer, in [m]')

    # Kalman Filter Settings
    parser.add_argument('--kf_type', type=str, default='IEKF', help='kalman filter type, format: IEKF or UKF')
    # dX = [ΔPos_North, ΔPos_East, ΔH_asl, ΔVel_North, ΔVel_East, ΔVel_Down, ΔAcc_North, ΔAcc_East, ΔAcc_Down,
    #                                                                           Δψ, Δθ, Δφ] space state vector
    parser.add_argument('--kf_state_size', type=int, default=12, help='number of state estimation')

    args = parser.parse_args()

    # Flight Settings
    if not args.traj_from_file:
        parser.add_argument('--init_lat', type=float, default=37.5, help='initial Latitude, in [deg]')
        parser.add_argument('--init_lon', type=float, default=21.5, help='initial Longitude, in [deg]')
        parser.add_argument('--init_height', type=float, default=5000, help='flight height at start, in [m]')
        #
        parser.add_argument('--avg_spd', type=float, default=250, help='flight average speed, [in m/sec]')
        parser.add_argument('--psi', type=float, default=45, help='Yaw at start, in [deg]')
        parser.add_argument('--theta', type=float, default=0, help='Pitch at start, in [deg]')
        parser.add_argument('--phi', type=float, default=0, help='Roll at start, in [deg]')
        #
        parser.add_argument('--acc_north', type=float, default=0, help='acceleration in the north - south at start, '
                                                                       'in [m/s^2]')
        parser.add_argument('--acc_east', type=float, default=0, help='acceleration in the east - west axis at start, '
                                                                      'in [m/s^2]')
        parser.add_argument('--acc_down', type=float, default=0, help='acceleration in vertical axis at start, '
                                                                      'in [m/s^2]')
        #
        parser.add_argument('--psi_dot', type=float, default=0, help='change in psi during flight, '
                                                                     'in [deg/s]')
        parser.add_argument('--theta_dot', type=float, default=0, help='change in theta during flight, '
                                                                       'in [deg/s]')
        parser.add_argument('--phi_dot', type=float, default=0, help='change in phi during flight, '
                                                                     'in [deg/s]')
    else:  # already read from file
        pass

    # Other Defaults
    args = parser.parse_args()
    args.run_points = int(args.time_end / args.time_res)
    args.time_vec = np.arange(args.time_init, args.time_end, args.time_res)
    args.map_res = 3 if args.map_level == 1 else 1
    args.results_folder = os.path.join(os.getcwd(), 'out')
    args.imu_errors = {
        'velocity': args.flg_err_vel * args.val_err_vel,
        'initial_position': args.flg_err_pos * args.val_err_pos,
        'euler_angles': args.flg_err_eul * args.val_err_eul,
        'barometer_bias': args.flg_err_baro_bias * args.val_err_baro_bias,
        'barometer_noise': args.flg_err_baro_noise * args.val_err_baro_noise,
        'altimeter_noise': args.flg_err_alt * args.val_err_alt,
    }

    return args


class Map:
    def __init__(self):
        self.meta = None
        self.axis = None
        self.bounds = None
        self.mpd = None
        self.grid = None

    def __str__(self):
        if self.meta is None:
            return 'Map Class instance: not initialized'
        else:
            map_level = self.meta['map_level']
            return f'Map class instance: map level {map_level}, {self.bounds}'

    def load(self, args):
        tile_length, map_level, ext = (1200, 1, 'dt1') if args.map_res == 3 else (3600, 3, 'dt2')
        self.meta = {
            'maps_dir': args.maps_dir,
            'map_res': args.map_res,
            'map_level': map_level,
            'rate': args.map_res / 3600,
            'tile_length': tile_length,
            'ext': ext,
            'out_folder': args.results_folder
        }
        self.meta['ext'] = 'mat'
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

        # Create an empty map grid
        if lat is not None and lon is not None:
            min_lat, max_lat = np.floor(lat), np.ceil(lat)
            min_lon, max_lon = np.floor(lon), np.ceil(lon)

        # set boundaries
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
            print(f'Trying to load tile: {tile_path}')
            ret = sp.loadmat(tile_path)['elevation_data']
            # ret = sp.loadmat(tile_path).get('data', np.zeros((tile_length + 1, tile_length + 1)))
            print(f'Loaded tile: E0{e} n{n}')
            return ret
        except FileNotFoundError:
            print(f'file not found: {tile_path}')
            return None

    def _validate_map(self, map_full_tiles):
        if np.all(map_full_tiles == 0) or np.all(np.isnan(map_full_tiles)):
            self.grid = mocking_map(map_full_tiles)
        else:
            self.grid = map_full_tiles.astype(int)

    #TODO: start from here, the update
    def update_map(self, new_lat, new_lon):
        new_args = set_settings()
        new_args.init_lat = new_lat, new_args.init_lon = new_lon
        new_map = Map().load(new_args)

        directions = []
        if new_lat > self.bounds['lat'][1]: directions.append('north')
        if new_lat < self.bounds['lat'][0]: directions.append('south')
        if new_lon < self.bounds['lon'][0]: directions.append('west')
        if new_lon > self.bounds['lon'][1]: directions.append('east')

        # Update map attributes for each necessary direction
        for direction in directions:
            self._update_map_attributes(new_map, direction)

    def _update_map_attributes(self, new_map, axis):
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
        except Exception as e:
            print(f"Failed to save the map: {e}")

    def visualize_map(self, mode, save=False):
        """
        Visualizes the map in either 2D or 3D mode.

        :param mode: A string that determines the visualization mode.
                     '2D' for a two-dimensional plot and '3D' for a three-dimensional plot.
        """
        if self.grid is None:
            print("Map grid is not initialized.")
            return

        if mode == '2D': # 2D visualization
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



class TrajFromFile(BaseTraj):
    pass
