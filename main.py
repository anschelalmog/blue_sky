from src.data_loaders import Map, set_settings
from src.create_traj import CreateTraj
from src.noise_traj import NoiseTraj
from src.estimators import IEKF, UKF
from src.outputs_utils import Errors, Covariances, plot_results

if __name__ == '__main__':
    args = set_settings()   # Set the system settings
    map_data = Map(args).load()      # Load the map data using the provided settings
    # Create the actual trajectory based on the map data and settings
    args.psi = 45
    true_traj = CreateTraj(args).linear(map_data)
    # if args.traj_type == 'linear': true_traj.linear(map_data)
    # else: true_traj.constant_acceleration(map_data)
    args.psi = 22
    true_traj_22 = CreateTraj(args).linear(map_data)

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
