"""
    Continuous Update Algorithm based on Kalman Filter
    ##################################################
    ALMOG ANSCHEL SEPTEMBER 2023
    ##################################################
"""
from modules.base_classes import Errors
from modules.data_loaders import LoadMap, TrajFromFile
from modules.generate_traj import CreateTraj, NoiseTraj
from modules.kalman_filters import IteratedExtendedKF, UnscentedKF
from modules.plot_run import plot_results
import argparse
import os
import pickle
import numpy as np


# import matplotlib.pyplot as plt


def retrieve_data_(function, filename, **kwargs):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        data = function(**kwargs)
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        return data


def parse_args():
    parser = argparse.ArgumentParser()

    # Run Settings
    parser.add_argument('--run_from_file', type=bool, default=False, help='Run from a file, or generate a trajectory')
    parser.add_argument('--run_anew', type=bool, default=True, help='Run from a file, or generate a trajectory')
    parser.add_argument('--run_file_path', type=str, default=' ', help='path to the run file')
    parser.add_argument('--plot_results', type=bool, default=True, help='Plots Results in the end of the run')

    # Map
    parser.add_argument('--map_path', type=str, default='D:\python_project\project_B\Map', help='folder path')
    parser.add_argument('--map_level', type=int, default=1, help='Map Level')

    # Flight
    parser.add_argument('--psi', type=int, default=45, help='Yaw at start, in [deg]')
    # parser.add_argument('--psi', type=int, default=45, help='Yaw at start, in [deg]')
    parser.add_argument('--theta', type=int, default=0, help='Pitch at start, in [deg]')
    parser.add_argument('--phi', type=int, default=0, help='Roll at start, in [deg]')
    parser.add_argument('--lat', type=int, default=31.5, help='Latitude at start, in [deg]')
    parser.add_argument('--lon', type=int, default=23.5, help='Longitude at start, in [deg]')
    parser.add_argument('--height', type=int, default=5000, help='height above sea level at start, in [m]')
    parser.add_argument('--avg_spd', type=int, default=250, help='flights average speed, in [m]')

    # Time
    parser.add_argument('--time_init', type=int, default=0, help='times starts counting, in [sec]')
    parser.add_argument('--time_end', type=int, default=100, help='times ends counting, in [sec]')
    parser.add_argument('--time_rate', type=float, default=1, help='time rate, in [sec]')
    parser.add_argument('--time_res', type=float, default=0.1, help='flights resolution speed, in [sec]')

    # Errors
    parser.add_argument('--err_vel', type=int, default=0, help='error for velocity, in [m/s]')
    parser.add_argument('--err_pos', type=int, default=0, help='error for position, in [m]')
    parser.add_argument('--err_alt', type=int, default=0, help='error for altimeter, in [m]')
    parser.add_argument('--err_euler', type=int, default=0, help='error for euler angels, in [deg]')
    parser.add_argument('--err_baro_noise', type=int, default=0, help='error for barometer, in [m]')
    parser.add_argument('--err_baro_bias', type=int, default=0, help='error for barometer, in [m]')

    # KF base
    parser.add_argument('--kf_type', type=str, default='IEKF', help='kalman filter type')
    # dX = [delP_North, delP_East, delH, delV_North, delV_East, delV_Down] , space state vector
    parser.add_argument('--kf_state_size', type=int, default=6, help='number of state estimation')

    # IEKF
    parser.add_argument('--iekf_iters', type=int, default=1, help='number of max iter per estimate cycle in IEKF')
    parser.add_argument('--iekf_conv_rate', type=float, default=0.0, help='sufficient convergence rate in IEKF')

    # UKF
    parser.add_argument('--ukf_alpha', type=float, default=1e-3, help='spread of sigma points around the mean:[1e-4,1]')
    parser.add_argument('--ukf_beta', type=float, default=2, help='UKF beta')
    parser.add_argument('--ukf_kappa', type=float, default=0, help='ukf state estimation or parameter estimation')

    args = parser.parse_args()

    # other arguments
    args.run_points = int(args.time_end / args.time_res)  # how many data points in this run
    args.time_vec = np.arange(args.time_init, args.time_end, args.time_res)
    args.map_res = 3 if args.map_level == 1 else 1  # Map Resolution
    args.results_folder = os.path.join(os.getcwd(), 'Results')

    args.kf_P_est0 = np.power(np.diag([200, 200, 30, 2, 2, 2]), 2)
    args.kf_Q = np.power(np.diag([0, 0, 0, 1, 1, 1]), 1e-6)

    """
    #  apply errors by hand
    args.err_vel, args.err_pos, args.err_alt, args.err_euler, args.err_baro_noise, err_baro_bias = 
    10, 500, 100, 0.01, 10, 30
    """
    return args


def main():
    args = parse_args()  # Set Constants, Default values and Initial conditions

    # run from file or generate traj
    if args.run_from_file:
        true_traj = None
        meas_traj, map_data = TrajFromFile(args)
    else:
        # map_data = retrieve_data_(LoadMap, 'map_data.pkl', args=args)
        # true_traj = retrieve_data_(CreateTraj, 'true_traj.pkl', args=args, map_data=map_data)
        # meas_traj = retrieve_data_(NoiseTraj, 'meas_traj.pkl', args=args, true_traj=true_traj)
        map_data = LoadMap(args)
        true_traj = CreateTraj(args, map_data)
        meas_traj = NoiseTraj(args, true_traj)

    # choose which kind of estimation
    if args.kf_type == 'IEKF':
        estimation_results = IteratedExtendedKF(args, map_data, meas_traj)
        # estimation_results = retrieve_data_(IteratedExtendedKF, 'est_results_iekf.pkl', args, map_data, meas_traj)
    else:
        estimation_results = UnscentedKF()  # args, map_data, meas_traj)

    # calc errors
    errors = Errors(args, meas_traj, estimation_results, true_traj)
    # errors._print()

    # plot results
    if args.plot_results:
        # configure which plots to run
        plots = {
            'plot map': True,
            'position errors': True,
            'velocity errors': True,
            'attitude errors': False,
            'altitude errors': True,
            'model errors': True,
            'kalman gains': True,
            'map elevation': True,
        }

        # plot
        plot_results(args, map_data, true_traj, meas_traj, estimation_results, errors, plots)


if __name__ == "__main__":
    main()

# TODO: check covariances update
# TODO: add a variable for est.map = est.asl - est.agl
# TODO: why dx[2] is 0?
# TODO: check Z, vary the model

#
# remark: when using flat surface all the kalman gains are 0, as formula
