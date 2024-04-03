import sys
import time
from matplotlib import pyplot as plt
import os

from src.data_loaders import Map, set_settings, IMUErrors
from src.create_traj import CreateTraj
from src.noise_traj import NoiseTraj
from src.estimators import IEKF, UKF
from src.outputs_utils import RunErrors, Covariances, plot_results, print_log

if __name__ == '__main__':
    args = set_settings()  # Set the system settings
    errors = IMUErrors(args.imu_errors)

    map_data = Map().load(args)  # Load the map data using the provided settings
    # map_data.visualize_map(mode='2D', save=False)

    # Create the actual trajectory based on the map data and settings
    true_traj = CreateTraj(args).create(map_data)

    # Generate a noisy trajectory to simulate the sensor measurements
    meas_traj = NoiseTraj(true_traj)
    meas_traj.add_noise(errors.imu_errors, dist=args.noise_type, approach='bottom-up')

    time.sleep(0.1)

    # estimation_results = UKF(args).run(map_data, meas_traj)

    # if args.kf_type == 'IEKF':
    # runs Iterated Extended Kalman Filter
    estimation_results = IEKF(args).run(map_data, meas_traj)
    # else:  # args.kf_type == 'UKF':
    # runs Unscented Kalman Filter
    #     estimation_results = UKF(args).run(map_data, meas_traj)

    args.plots = {
        'plot map': True,
        'position errors': True,
        'velocity errors': True,
        'attitude errors': True,
        'altitude errors': True,
        'model errors': True,
        'kalman gains': True,
        'map elevation': True,
    }

    used_traj = true_traj if true_traj is not None else meas_traj
    errors = RunErrors(used_traj, estimation_results.traj)
    covariances = Covariances(estimation_results.params.P_est)

    plot_results(args, map_data, true_traj, meas_traj, estimation_results, errors, covariances)
    print_log(args, estimation_results, errors, covariances)

"""
px4 flight sunykatir  user guide toturial ,

Qground control
JSBC --- מקשרת בין שתיהן
FLight Control
gzibo
"""
