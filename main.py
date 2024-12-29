import BLUE_SKY as BS

if __name__ == '__main__':
    # Set the system settings
    args = BS.set_settings()

    errors = BS.IMUErrors(args.imu_errors)

    # Load the map data using the provided settings
    # Create the map class with the elevation data
    map_data = BS.Map().load(args, mock=False)

    # Create the actual trajectory based on the map data and settings
    true_traj = BS.CreateTraj(args).create(map_data)

    # Generate a noisy trajectory to simulate the sensor measurements
    meas_traj = BS.NoiseTraj(true_traj).add_noise(errors.imu_errors, approach='top-down')

    # Perform trajectory estimation based on the selected Kalman filter type
    estimation_results = {'IEKF': BS.IEKF, 'UKF': BS.UKF}[args.kf_type](args).run(map_data, meas_traj)

    # Calculate errors and covariances between the used trajectory and the estimated trajectory
    run_errors, covariances = BS.calc_errors_covariances(true_traj or meas_traj, estimation_results)

    # Plot the results based on the plotting options
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
    BS.plot_results(args, map_data, true_traj, meas_traj, estimation_results, run_errors, covariances)

    # Print log information including settings, estimation results, errors, and covariances
    # print_log(args, estimation_results, run_errors, covariances)

#    # errors.set_imu_error('altimeter', amplitude=5)
    # errors.set_imu_error('barometer', amplitude=50)
    # errors.set_imu_error('velocity meter', amplitude=3)
    # errors.set_imu_error('gyroscope', amplitude=0.1)
