import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from modules.data_loaders import Map
from modules.create_traj import CreateTraj
from modules.noise_traj import NoiseTraj
from modules.estimators import IEKF, UKF
from modules.outputs_utils import Errors, Covariances, plot_results, print_log


def plot_trajectory_on_map(map_data, trajectory):
    plt.figure(figsize=(8, 8))
    plt.imshow(map_data.grid, cmap='terrain', extent=[map_data.lon_bounds[0], map_data.lon_bounds[1],
                                                      map_data.lat_bounds[0], map_data.lat_bounds[1]])
    plt.plot(trajectory.pos.lon, trajectory.pos.lat, color='red', linewidth=2)
    plt.title('Trajectory on Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()


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
    parser.add_argument('--flg_err_pos', type=bool, default=0, help='flag error for position')
    parser.add_argument('--flg_err_vel', type=bool, default=0, help='flag for error for velocity')
    parser.add_argument('--flg_err_alt', type=bool, default=0, help='flag error for altimeter')
    parser.add_argument('--flg_err_eul', type=bool, default=0, help='flag error for euler angels')
    parser.add_argument('--flg_err_baro_noise', type=bool, default=0, help='flag error for barometer noise')
    parser.add_argument('--flg_err_baro_bias', type=bool, default=0, help='flag error for barometer bias')

    # Errors Values
    parser.add_argument('--val_err_pos', type=int, default=200, help='error for position, in [m]')
    parser.add_argument('--val_err_vel', type=int, default=2, help='error for velocity, in [m/s]')
    parser.add_argument('--val_err_alt', type=int, default=5, help='error for altimeter, in [m]')
    parser.add_argument('--val_err_eul', type=int, default=0.05, help='error for euler angels, in [deg]')
    parser.add_argument('--val_err_baro_noise', type=int, default=5, help='error for barometer, in [m]')
    parser.add_argument('--val_err_baro_bias', type=int, default=5, help='error for barometer, in [m]')

    # Kalman Filter Settings 
    parser.add_argument('--kf_type', type=str, default='IEKF', help='kalman filter type, format: IEKF or UKF')
    # dX = [delP_North, delP_East, delH, delV_North, delV_East, delV_Down] , space state vector
    parser.add_argument('--kf_state_size', type=int, default=6, help='number of state estimation')

    args = parser.parse_args()
    # Flight Settings
    if not args.traj_from_file:
        args.init_lat = 31.5  # initial Latitude, in [deg]
        args.init_lon = 23.5  # initial Longitude, in [deg]
        args.init_height = 5000  # flight height at start, in [m]
        args.avg_spd = 250  # flight average speed, [in m/sec]
        args.psi = 22.5  # Yaw at start, in [deg]
        args.theta = 0  # Pitch at start, in [deg]
        args.phi = 0  # Roll at start, in [deg]
    else:
        pass

    # Other Defaults
    args.run_points = int(args.time_end / args.time_res)
    args.time_vec = np.arange(args.time_init, args.time_end, args.time_res)
    args.map_res = 3 if args.map_level == 1 else 1
    args.results_folder = os.path.join(os.getcwd(), 'Results')
    args.imu_errors = {
        'velocity': args.flg_err_vel * args.val_err_vel,
        'initial_position': args.flg_err_pos * args.val_err_pos,
        'euler_angles': args.flg_err_eul * args.val_err_eul,
        'barometer_bias': args.flg_err_baro_bias * args.val_err_baro_bias,
        'barometer_noise': args.flg_err_baro_noise * args.val_err_baro_noise,
        'altimeter_noise': args.flg_err_alt * args.val_err_alt,
    }

    return args


def main():
    args = set_settings()

    map_data = Map(args).load()  # load map
    true_traj = CreateTraj(args).create_linear(map_data)  # where we actually are
    meas_traj = NoiseTraj(true_traj).noise(args.imu_errors, dist=args.noise_type)  # where we think we are
    time.sleep(0.1)
    # plot_trajectory_on_map(map_data, meas_traj)

    if args.kf_type == 'IEKF':
        # runs Iterated Extended Kalman Filter
        estimation_results = IEKF(args).run(map_data, meas_traj)
    else:  # args.kf_type == 'UKF':
        # runs Unscented Kalman Filter
        estimation_results = UKF(args).run(map_data, meas_traj)

    # configure which plots to run
    args.plots = {
        'plot map': True,
        'position errors': True,
        'velocity errors': True,
        'attitude errors': False,
        'altitude errors': True,
        'model errors': True,
        'kalman gains': True,
        'map elevation': True,
    }

    used_traj = true_traj if true_traj is not None else meas_traj
    errors = Errors(used_traj, estimation_results.traj)
    covariances = Covariances(estimation_results.params.P_est)

    plot_results(args, map_data, true_traj, meas_traj, estimation_results, errors, covariances)
    pass
    # print_log(args, estimation_results.params, errors, covariances)


if __name__ == '__main__':
    main()
