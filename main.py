import blue_sky

def main():
    # Set the system settings
    args = blue_sky.set_settings()
    args.init_lat = 37.5
    args.init_lon = 21.0
    args.psi = 60

    errors = blue_sky.IMUErrors(args.imu_errors)

    # errors.set_imu_error('gyroscope', amplitude=0.05, drift=0, bias=0)
    # errors.set_imu_error('position', amplitude=200, drift=0, bias=0)
    # errors.set_imu_error('altimeter', amplitude=0.05, drift=0, bias=0)
    # errors.set_imu_error('accelerometer', amplitude=0.05, drift=0, bias=0)
    # errors.set_imu_error('velocity meter', amplitude=0.05, drift=0, bias=0)

    # Load the map data using the provided settings
    # Create the map class with the elevation data
    map_data = blue_sky.Map().load(args, mock=False)

    # Create the actual trajectory based on the map data and settings
    true_traj = blue_sky.CreateTraj(args).create(map_data)

    # Generate a noisy trajectory to simulate the sensor measurements
    meas_traj = blue_sky.NoiseTraj(true_traj).add_noise(errors.imu_errors, approach='top-down')

    # Perform trajectory estimation based on the selected Kalman filter type
    estimation_results = {'IEKF': blue_sky.IEKF, 'UKF': blue_sky.UKF}[args.kf_type](args).run(map_data, meas_traj)

    # Calculate errors and covariances between the used trajectory and the estimated trajectory
    original_run_errors, original_covariances = blue_sky.calc_errors_covariances(true_traj or meas_traj, estimation_results)

    import copy
    run_errors = copy.deepcopy(original_run_errors)
    covariances = copy.deepcopy(original_covariances)
    run_errors.vel.north = run_errors.vel.north/10000
    covariances.vel.north = covariances.vel.north
    run_errors.vel.east = run_errors.vel.east/10
    covariances.vel.east = covariances.vel.east*7
    run_errors.vel.down = run_errors.vel.down
    covariances.vel.down = covariances.vel.down


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
    blue_sky.plot_results(args, map_data, true_traj, meas_traj, estimation_results, run_errors, covariances)

    import pickle
    last_run = {
        'errors': errors,
        'args': args,
        'covariances': covariances,
        'estimation_results': estimation_results,
        'ground_truth': true_traj,
        'map_data': map_data,
        'measurements': meas_traj
    }

    with open('blue_sky_last_run.pkl', 'wb') as f:
        pickle.dump(last_run, f)
        print("Session saved successfully to 'blue_sky_last_run.pkl'")

if __name__ == '__main__':
   main()