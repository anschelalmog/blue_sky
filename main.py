from src.data_loaders import Map, set_settings
from src.create_traj import CreateTraj
from src.noise_traj import NoiseTraj
from src.estimators import IEKF, UKF
from src.outputs_utils import Errors, Covariances, plot_results

# for debug
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    args = set_settings()  # Set the system settings
    map_data = Map(args).load()  # Load the map data using the provided settings
    # Create the actual trajectory based on the map data and settings
    true_traj = CreateTraj(args).linear(map_data)
    # Generate a noisy trajectory to simulate the sensor measurements
    meas_traj = NoiseTraj(true_traj).noise(args.imu_errors, dist=args.noise_type)

    if args.kf_type == 'IEKF':
        # runs Iterated Extended Kalman Filter
        estimation_results = IEKF(args).run(map_data, meas_traj)
    else:  # args.kf_type == 'UKF':
        # runs Unscented Kalman Filter
        estimation_results = UKF(args).run(map_data, meas_traj)

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

# if args.traj_type == 'linear': true_traj.linear(map_data)
# else: true_traj.constant_acceleration(map_data)

"""
    args.psi = 22
    true_traj_22 = CreateTraj(args).linear(map_data)
    meas_traj_22 = NoiseTraj(true_traj).noise(args.imu_errors, dist=args.noise_type)
    estimation_results_22 = IEKF(args).run(map_data, meas_traj)

    args.psi = 0
    true_traj_0 = CreateTraj(args).linear(map_data)
    meas_traj_0 = NoiseTraj(true_traj).noise(args.imu_errors, dist=args.noise_type)
    estimation_results_0 = IEKF(args).run(map_data, meas_traj)

    breakpoint()
    t = args.time_vec

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    title = 'Lat Lon Errors DEBUG'
    fig.suptitle(title, fontsize=24, fontweight='bold')

    axs[0].plot(t, true_traj.pos.lat - true_traj_22.pos.lat, '-r', linewidth=1)
    axs[0].plot(t, true_traj.pos.lat - true_traj_0.pos.lat, '-b', linewidth=1)
    axs[0].plot(t, true_traj_0.pos.lat - true_traj_22.pos.lat, '-g', linewidth=1)
    axs[0].set_title('lat errors')
    axs[0].set_xlabel('Time [sec]')
    axs[0].grid(True)
    axs[0].legend(['45 - 22', '45 - 0', '0 - 22'], loc='best')

    axs[1].plot(t, true_traj.pos.lon - true_traj_22.pos.lon, '-r', linewidth=1)
    axs[1].plot(t, true_traj.pos.lon - true_traj_0.pos.lon, '-b', linewidth=1)
    axs[1].plot(t, true_traj_0.pos.lon - true_traj_22.pos.lon, '-g', linewidth=1)
    axs[1].set_xlabel('Time [sec]')
    axs[1].grid(True)
    axs[1].legend(['45 - 22', '45 - 0', '0 - 22'], loc='best')
    plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
    plt.show()

    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    title = 'North East Errors DEBUG'
    fig.suptitle(title, fontsize=24, fontweight='bold')
    axs[0].plot(t, true_traj.pos.north - true_traj_22.pos.north, '-r', linewidth=1)
    axs[0].plot(t, true_traj.pos.north - true_traj_0.pos.north, '-b', linewidth=1)
    axs[0].plot(t, true_traj_0.pos.north - true_traj_22.pos.north, '-g', linewidth=1)
    axs[0].set_title('north errors')
    axs[0].set_xlabel('Time [sec]')
    axs[0].grid(True)
    axs[0].legend(['45 - 22', '45 - 0', '0 - 22'], loc='best')

    axs[1].plot(t, true_traj.pos.east - true_traj_22.pos.east, '-r', linewidth=1)
    axs[1].plot(t, true_traj.pos.east - true_traj_0.pos.east, '-b', linewidth=1)
    axs[1].plot(t, true_traj_0.pos.east - true_traj_22.pos.east, '-g', linewidth=1)
    axs[1].set_xlabel('Time [sec]')
    axs[1].grid(True)
    axs[1].legend(['45 - 22', '45 - 0', '0 - 22'], loc='best')
    plt.savefig(os.path.join(args.results_folder, f'{title}.png'))
    plt.show()
"""
"""
    debug
"""