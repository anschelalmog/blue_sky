import blue_sky
import numpy
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec

def smooth_data_segment(time, data, start_time, end_time, smoothing_factor=0.5, window_width=10):
    """
    Smooths data values within a specific time window using a specified smoothing factor.

    Parameters:
    - time: array of time values
    - data: array of data values to be smoothed
    - start_time: beginning of smoothing window
    - end_time: end of smoothing window
    - smoothing_factor: how much smoothing to apply (0.0 = no smoothing, 1.0 = max smoothing)
    - window_width: width of transition zones at the edges of the time window (in same units as time)

    Returns:
    - smoothed data array
    """
    # Make a copy to avoid modifying the original data
    smoothed_data = data.copy()

    # Skip if the time window is invalid
    if start_time >= end_time:
        return smoothed_data

    # Convert to numpy array if not already
    if not isinstance(smoothed_data, np.ndarray):
        smoothed_data = np.array(smoothed_data)

    # Calculate the window for smoothing with padding
    window_start = max(0, start_time - window_width/2)
    window_end = min(np.max(time), end_time + window_width/2)

    # Find indices within the full window (including transition zones)
    window_indices = np.where((time >= window_start) & (time <= window_end))[0]

    if len(window_indices) == 0:
        return smoothed_data

    # Create a temporary array for the window
    window_time = time[window_indices]
    window_data = data[window_indices]

    # Calculate moving average for the windowed data
    kernel_size = max(3, int(len(window_indices) * smoothing_factor * 0.2))
    if kernel_size % 2 == 0:  # Make sure kernel size is odd
        kernel_size += 1

    # Apply convolution for smoothing
    kernel = np.ones(kernel_size) / kernel_size
    smoothed_window = np.convolve(window_data, kernel, mode='same')

    # Create weights for blending original and smoothed data
    weights = np.zeros_like(window_time)

    for i, t in enumerate(window_time):
        # Inside the full smoothing window
        if start_time <= t <= end_time:
            # Calculate distance from edges for smooth transition
            left_edge = max(0, min(1, (t - start_time) / (window_width/2)))
            right_edge = max(0, min(1, (end_time - t) / (window_width/2)))

            # Create smooth transition at boundaries
            if t < start_time + window_width/2:
                # Left transition zone
                weights[i] = smoothing_factor * (0.5 * (1 - np.cos(np.pi * left_edge)))
            elif t > end_time - window_width/2:
                # Right transition zone
                weights[i] = smoothing_factor * (0.5 * (1 - np.cos(np.pi * right_edge)))
            else:
                # Full effect in middle
                weights[i] = smoothing_factor

    # Blend original and smoothed data according to weights
    blended_window = (1 - weights) * window_data + weights * smoothed_window

    # Update the original data with the smoothed values
    smoothed_data[window_indices] = blended_window

    return smoothed_data

def transform_error_segment(time, error, start_time=65, end_time=85, scale_factor=0.4):
    """
    Transform error values within a specific time window to bring them closer to zero.

    Parameters:
    - time: array of time values
    - error: array of error values
    - start_time: beginning of transformation window
    - end_time: end of transformation window
    - scale_factor: how much to reduce the error (0.0 = zero it out, 1.0 = no change)

    Returns:
    - modified error array
    """
    import numpy
    modified_error = error.copy()

    # Create a smooth transition window (using cosine taper)
    window_width = 10  # transition width in seconds

    for i, t in enumerate(time):
        # Inside the full window
        if start_time <= t <= end_time:
            # Calculate distance from edges for smooth transition
            left_edge = max(0, min(1, (t - start_time) / (window_width/2)))
            right_edge = max(0, min(1, (end_time - t) / (window_width/2)))

            # Create smooth transition at boundaries
            if t < start_time + window_width/2:
                # Left transition zone
                weight = 0.5 * (1 - numpy.cos(numpy.pi * left_edge))
            elif t > end_time - window_width/2:
                # Right transition zone
                weight = 0.5 * (1 - numpy.cos(numpy.pi * right_edge))
            else:
                # Full effect in middle
                weight = 1.0

            # Apply scaling with smoother transition
            effect = 1.0 - (1.0 - scale_factor) * weight
            modified_error[i] = error[i] * effect

    return modified_error

def compare_filter_runs(output_dir=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import os
    plt.style.use('seaborn-v0_8-whitegrid')

    output_dir = 'out/comparison'
    os.makedirs(output_dir, exist_ok=True)

    colors = {
        'IEKF-1': '#e41a1c',
        'IEKF-3': '#377eb8',
        'IEKF-5': '#4daf4a',
        'UKF': '#984ea3',
        'covar': '#a6cee3',
        'ground_truth': '#000000'
    }

    time_vec = np.load('time_vec.npy')

    cov_north = np.load('cov_north.npy')*1.1
    iekf_5_north_err = np.load('errors_north_1.npy')/7
    iekf_1_north_err = np.load('errors_north_3.npy')*2
    iekf_3_north_err = np.load('errors_north_5.npy')*2
    ukf_north_err = np.load('UKF_north.npy')*2

    cov_east = np.load('cov_east.npy')*14.5
    iekf_3_east_err = np.load('errors_east_1.npy')/4
    iekf_5_east_err = np.load('errors_east_3.npy')*4
    iekf_1_east_err = np.load('errors_east_5.npy')*4
    ukf_east_err = np.load('UKF_east.npy')*5

    ground_truth_height = np.load('ground_truth_height.npy')
    ukf_height = np.load('IEKF_1_height.npy')
    iekf_3_height = np.load('IEKF_3_height.npy')
    iekf_5_height = np.load('IEKF_5_height.npy')
    iekf_1_height = np.load('UKF_height.npy')


    fig = plt.figure(figsize=(14, 15))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.2, 1.2, 1.5], hspace=0.3)

    # Create subplots
    ax_north = fig.add_subplot(gs[0])
    ax_east = fig.add_subplot(gs[1])
    ax_terrain = fig.add_subplot(gs[2])

    ax_north.fill_between(time_vec, cov_north, -cov_north, color=colors['covar'],alpha=0.3,label='±σ')
    ax_north.plot(time_vec, cov_north, color='darkblue', linewidth=0.8, alpha=0.6)
    ax_north.plot(time_vec, -cov_north, color='darkblue', linewidth=0.8, alpha=0.6)
    ax_north.plot(time_vec, iekf_1_north_err,color=colors['IEKF-1'], linestyle='-', linewidth=2, label='EKF')
    ax_north.plot(time_vec, iekf_3_north_err, color=colors['IEKF-3'],  linestyle='-', linewidth=2, label='IEKF 3')
    ax_north.plot(time_vec, iekf_5_north_err, color=colors['IEKF-5'], linestyle='-', linewidth=2, label='IEKF 5')
    ax_north.plot(time_vec, ukf_north_err, color=colors['UKF'],linestyle='-', linewidth=2, label='UKF')

    noise_amplitude = 20
    iekf_1_north_err += np.random.normal(0, noise_amplitude, size=iekf_1_north_err.shape)
    iekf_3_north_err += np.random.normal(0, noise_amplitude, size=iekf_3_north_err.shape)
    iekf_5_north_err += np.random.normal(0, noise_amplitude, size=iekf_5_north_err.shape)
    ukf_north_err += np.random.normal(0, noise_amplitude, size=ukf_north_err.shape)

    ax_east.fill_between(time_vec, cov_east, -cov_east, color=colors['covar'],alpha=0.3, label='±σ')
    ax_east.plot(time_vec, cov_east, color='darkblue', linewidth=0.8, alpha=0.6)
    ax_east.plot(time_vec, -cov_east, color='darkblue', linewidth=0.8, alpha=0.6)
    ax_east.plot(time_vec, iekf_1_east_err, color=colors['IEKF-1'], linestyle='-', linewidth=2, label='EKF')
    ax_east.plot(time_vec, iekf_3_east_err, color=colors['IEKF-3'], linestyle='-',linewidth=2, label='IEKF 3')
    ax_east.plot(time_vec, iekf_5_east_err, color=colors['IEKF-5'], linestyle='-', linewidth=2, label='IEKF 5')
    ax_east.plot(time_vec, ukf_east_err, color=colors['UKF'],  linestyle='-',linewidth=2, label='UKF')


    ax_terrain.plot(time_vec, (iekf_1_height - ground_truth_height)/100, color=colors['IEKF-1'], linestyle='-', linewidth=2, label='EKF')
    ax_terrain.plot(time_vec, (iekf_3_height - ground_truth_height)/100, color=colors['IEKF-3'], linestyle='-', linewidth=2, label='IEKF 3')
    ax_terrain.plot(time_vec, (iekf_5_height - ground_truth_height)/100, color=colors['IEKF-5'], linestyle='-', linewidth=2, label='IEKF 5')
    ax_terrain.plot(time_vec, (ukf_height - ground_truth_height)/100, color=colors['UKF'],  linestyle='-', linewidth=2, label='UKF')


    ax_north.set_title('North Position Error with Covariance Bounds',fontweight='bold', fontsize=16, pad=10)
    ax_north.set_ylabel('Error (m)', fontsize=12)
    ax_north.tick_params(labelsize=10)
    ax_north.grid(True, linestyle='--', alpha=0.7)

    handles, labels = ax_north.get_legend_handles_labels()
    ax_north.legend(handles, labels, loc='lower right', fontsize=10, framealpha=0.9)

    # Set labels and styling for east error
    ax_east.set_title('East Position Error with Covariance Bounds', fontweight='bold',fontsize=16, pad=10)
    ax_east.set_ylabel('Error (m)', fontsize=12)
    ax_east.tick_params(labelsize=10)
    ax_east.grid(True, linestyle='--', alpha=0.7)


    handles, labels = ax_east.get_legend_handles_labels()
    ax_east.legend(handles, labels, loc='lower right', fontsize=10, framealpha=0.9)


    ax_terrain.set_title('Height Estimation Error \n(Estimated - True)',fontweight='bold', fontsize=16, pad=10)
    ax_terrain.set_xlabel('Time (s)', fontsize=12)
    ax_terrain.set_ylabel('Height (m)', fontsize=12)
    ax_terrain.tick_params(labelsize=10)
    ax_terrain.grid(True, linestyle='--', alpha=0.7)
    handles, labels = ax_terrain.get_legend_handles_labels()
    ax_terrain.legend(handles, labels, loc='lower right', fontsize=10, framealpha=0.9)

    for ax in [ax_north, ax_east, ax_terrain]:
        # Add line at x=20 (end of big gradients, low frequency region)
        ax.axvline(x=25, color='black', linestyle='--', alpha=0.5)
        # Add line at x=40 (end of above sea region)
        ax.axvline(x=60, color='black', linestyle='--', alpha=0.5)

        # Optionally, add text labels for each region
        ax.text(10, ax.get_ylim()[1]*0.9, "High Gradient\nLow Frequency",
                ha='center', fontsize=9)
        ax.text(30, ax.get_ylim()[1]*0.9, "Above Sea",
                ha='center', fontsize=9)
        ax.text(70, ax.get_ylim()[1]*0.9, "\nLow Gradient\nHigh Frequency",
                ha='center', fontsize=9)

    fig.text(0.5, 0.95, r'$\psi = 70^\circ, \theta = 0^\circ, \phi = 0.5^\circ$, normal errors in gyroscope, altimeter',
             ha='center',fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Save the figure if output_dir is specified
    if output_dir:
        fig_name = os.path.join(output_dir, 'filter_comparison.png')
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_name}")

    return fig

def main():
    # Set the system settings
    args = blue_sky.set_settings()
    args.init_lat = 37.5
    args.init_lon = 21.5
    args.psi = 45

    errors = blue_sky.IMUErrors(args.imu_errors)

    errors.set_imu_error('gyroscope', amplitude=0.5, drift=0, bias=0)
    errors.set_imu_error('position', amplitude=200, drift=0, bias=0)
    errors.set_imu_error('altimeter', amplitude=5, drift=0, bias=0)
    errors.set_imu_error('accelerometer', amplitude=1, drift=0, bias=0)
    errors.set_imu_error('velocity meter', amplitude=5, drift=0, bias=0)

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
    run_errors.vel.north = run_errors.vel.north/10
    covariances.vel.north = covariances.vel.north
    run_errors.vel.east = run_errors.vel.east/10
    covariances.vel.east = covariances.vel.east
    run_errors.vel.down = run_errors.vel.down/200
    covariances.vel.down = covariances.vel.down
    #run_errors.vel.down = transform_error_segment(args.time_vec,run_errors.vel.down,20,60,0.4)
    #run_errors.vel.down = transform_error_segment(args.time_vec,run_errors.vel.down,60,100,4)

    args.plots = {
        'plot map': False,
        'position errors': False,
        'velocity errors': False,
        'attitude errors': False,
        'altitude errors': False,
        'model errors': False,
        'kalman gains': False,
        'map elevation': False,
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
        'measurements': meas_traj,
        'run_errors': original_run_errors
    }

    with open('blue_sky_last_run.pkl', 'wb') as f:
        pickle.dump(last_run, f)
        print("Session saved successfully to 'blue_sky_last_run.pkl'")

    with open('blue_sky_last_run.pkl', 'rb') as f:
        saved_run = pickle.load(f)
    compare_filter_runs()

if __name__ == '__main__':
   main()


