import sys
import time
from matplotlib import pyplot as plt
import os
import numpy as np
import itertools

from src.data_loaders import Map, set_settings, IMUErrors
from src.create_traj import CreateTraj
from src.noise_traj import NoiseTraj
from src.estimators import IEKF, UKF
from src.outputs_utils import RunErrors, Covariances, plot_results, print_log


def compare_trajectories(true_traj, meas_traj):
    print("Comparing Trajectory Components:")
    print("=" * 40)

    for attr in ['pos', 'vel', 'acc', 'euler']:
        true_obj = getattr(true_traj, attr)
        meas_obj = getattr(meas_traj, attr)

        print(f"Comparing {attr}:")
        for sub_attr in vars(true_obj):
            true_val = getattr(true_obj, sub_attr)
            meas_val = getattr(meas_obj, sub_attr)

            if not np.allclose(true_val, meas_val):
                print(f"  - {sub_attr} is not equal")
            else:
                print(f"  - {sub_attr} is equal")

        print()

    # Compare mpd_north and mpd_east
    print("Comparing mpd_north and mpd_east:")
    if not np.allclose(true_traj.mpd_north, meas_traj.mpd_north):
        print("  - mpd_north is not equal")
    else:
        print("  - mpd_north is equal")

    if not np.allclose(true_traj.mpd_east, meas_traj.mpd_east):
        print("  - mpd_east is not equal")
    else:
        print("  - mpd_east is equal")

    print("=" * 40)


def plot_height_profiles(create_traj, noise_traj):
    plt.figure(figsize=(10, 5))
    plt.title('Height Profile Comparison')
    plt.plot(create_traj.time_vec, create_traj.pos.h_agl, 'g-', label='True Height')
    plt.plot(noise_traj.time_vec, noise_traj.pos.h_agl, 'm--', label='Noisy Height')
    plt.xlabel('Time [s]')
    plt.ylabel('Height [m]')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_trajectory_comparison(create_traj, noise_traj, map_data):
    fig = plt.figure(figsize=(14, 7))

    # 2D Plot
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('2D Trajectory Comparison')
    X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
    ax1.contourf(X, Y, map_data.grid, cmap='terrain', alpha=0.7)
    ax1.plot(create_traj.pos.lon, create_traj.pos.lat, 'r-', label='True Trajectory')
    ax1.plot(noise_traj.pos.lon, noise_traj.pos.lat, 'b--', label='Noisy Trajectory')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend()

    # 3D Plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.title.set_text('3D Trajectory Comparison')
    surf = ax2.plot_surface(X, Y, map_data.grid.T, cmap='terrain', alpha=0.5)
    ax2.plot(create_traj.pos.lon, create_traj.pos.lat, create_traj.pos.h_asl, 'r-', label='True Trajectory')
    ax2.plot(noise_traj.pos.lon, noise_traj.pos.lat, noise_traj.pos.h_asl, 'b--', label='Noisy Trajectory')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_zlabel('Altitude')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = set_settings()  # Set the system settings
    errors = IMUErrors(args.imu_errors)

    # Load the map data using the provided settings
    map_data = Map().load(args)
    # map_data.visualize_map(mode='2D', save=False)

    # Create the actual trajectory based on the map data and settings
    true_traj = CreateTraj(args).create(map_data)

    # Generate a noisy trajectory to simulate the sensor measurements
    meas_traj = NoiseTraj(true_traj)
    meas_traj.add_noise(errors.imu_errors, dist=args.noise_type, approach='top-down')

    # plot_height_profiles(true_traj, meas_traj)
    # plot_trajectory_comparison(true_traj, meas_traj, map_data)
    # compare_trajectories(true_traj, meas_traj)

    time.sleep(0.1)

    ####################################################

    Q_values = [10 ** x for x in range(-2, 1)]

    # Define state components as a dictionary with initial best Q values
    state_components = {
        'pos_north': {'Q': None, 'rmse': float('inf')},
        'pos_east': {'Q': None, 'rmse': float('inf')},
        'pos_h_asl': {'Q': None, 'rmse': float('inf')},
        'vel_north': {'Q': None, 'rmse': float('inf')},
        'vel_east': {'Q': None, 'rmse': float('inf')},
        'vel_down': {'Q': None, 'rmse': float('inf')},
        'euler_psi': {'Q': None, 'rmse': float('inf')},
        'euler_theta': {'Q': None, 'rmse': float('inf')},
        'euler_phi': {'Q': None, 'rmse': float('inf')}
    }

    # Print the total number of combinations
    total_combinations = len(Q_values) ** len(state_components)
    print(f"Total number of Q combinations: {total_combinations}")

    single_iteration_duration = 2.5  # minutes
    expected_end_time = time.strftime("%H:%M %d-%m-%Y",
                                      time.localtime(
                                          time.time() + (total_combinations * single_iteration_duration * 60)))
    print(f"Expected end time: {expected_end_time}")

    # Loop over all Q combinations
    for idx, Q_combination in enumerate(itertools.product(Q_values, repeat=len(state_components))):
        Q = np.diag(Q_combination)  # Construct Q matrix from combination

        print(f"\nRunning IEKF with Q={Q_combination} (Combination {idx + 1}/{total_combinations})")
        start_time = time.time()  # Start the timer

        # Perform Kalman filter with current Q
        estimation_results = IEKF(args).run(map_data, meas_traj, Q)

        end_time = time.time()  # Stop the timer
        duration = end_time - start_time  # Calculate the duration

        used_traj = true_traj if true_traj is not None else meas_traj
        errors = RunErrors(used_traj, estimation_results.traj)
        covariances = Covariances(estimation_results.params.P_est)

        metrics = covariances.metrics

        # Update best Q values for each state component
        for component, best_Q_info in state_components.items():
            component_parts = component.split('_')
            component_type = component_parts[0]
            component_name = '_'.join(component_parts[1:])

            rmse_key = 'rmse'
            if component_type in metrics and component_name in metrics[component_type]:
                rmse_value = metrics[component_type][component_name][rmse_key]
                if rmse_value < best_Q_info['rmse']:
                    state_components[component] = {'Q': Q_combination, 'rmse': rmse_value}
                    print("best Q has changed")

            # Print the duration of the iteration
        print(f"Duration of iteration with Q={Q_combination}: {duration:.2f} seconds")

    # Print the best Q values for each state component
    print("\nBest Q values for each state component:")
    for component, best_Q in state_components.items():
        print(f"{component}: Q={best_Q['Q']}, RMSE={best_Q['rmse']:.4f}")

############################################

#########################################################################
    # if args.kf_type == 'IEKF':
    #     # runs Iterated Extended Kalman Filter
    #     estimation_results = IEKF(args).run(map_data, meas_traj)
    # else:  # args.kf_type == 'UKF':
    #     # runs Unscented Kalman Filter
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
