from src.data_loaders import *
from src.create_traj import *
from src.noise_traj import *
from src.estimators import *
from src.output_utils import *

if __name__ == '__main__':
    args = set_settings()  # Set the system settings
    errors = IMUErrors(args.imu_errors)
    #errors.set_imu_error('altimeter', amplitude=20)
    # errors.set_imu_error('barometer', amplitude=200, drift=0, bias=0)

    # Load the map data using the provided settings
    map_data = Map().load(args, mock=False)

    # Create the actual trajectory based on the map data and settings
    true_traj = CreateTraj(args).create(map_data)

    # Generate a noisy trajectory to simulate the sensor measurements
    meas_traj = NoiseTraj(true_traj).add_noise(errors.imu_errors)

    # Perform trajectory estimation based on the selected Kalman filter type
    if args.kf_type == 'IEKF':
        # runs Iterated Extended Kalman Filter
        estimation_results = IEKF(args).run(map_data, meas_traj)
    else:  # args.kf_type == 'UKF':
        # runs Unscented Kalman Filter
        estimation_results = UKF(args).run(map_data, meas_traj)

    # Calculate errors and covariances between the used trajectory and the estimated trajectory
    run_errors, covariances = calc_errors_covariances(true_traj or meas_traj, estimation_results)

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
    # Plot the results based on the plotting options
    plot_results(args, map_data, true_traj, meas_traj, estimation_results, run_errors, covariances)

    # Print log information including settings, estimation results, errors, and covariances
    # print_log(args, estimation_results, run_errors, covariances)
