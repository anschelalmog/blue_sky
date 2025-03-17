"""
Kalman Filter Comparison Tool for Terrain-Referenced Navigation

This script compares the performance of different Kalman filter configurations:
- IEKF with 1 iteration
- IEKF with 3 iterations
- IEKF with 5 iterations
- UKF

Results are saved as plots for presentation purposes with comparative metrics including CEP50.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import blue_sky
from tqdm import tqdm

def setup_experiment():
    """
    Set up the experiment with proper settings and create required directories.
    """
    # Create results directory
    results_dir = os.path.join(os.getcwd(), 'filter_comparison_results')
    os.makedirs(results_dir, exist_ok=True)

    # Get default settings
    args = blue_sky.set_settings()

    # Configure settings for comparison
    args.results_folder = results_dir
    args.plot_results = False  # We'll handle plotting ourselves

    # Set position error to make the comparison more interesting
    errors = blue_sky.IMUErrors(args.imu_errors)
    errors.set_imu_error('position', amplitude=20, drift=0, bias=0)

    return args, errors, results_dir

def run_filter(args, filter_type, iterations=None):
    """
    Run a specific filter configuration and return results.

    Args:
        args: Configuration arguments
        filter_type: Type of filter ('IEKF' or 'UKF')
        iterations: Number of iterations for IEKF (None for UKF)

    Returns:
        dict: Results of the filter run
    """
    # Make a copy of the arguments to avoid modifying the original
    run_args = deepcopy(args)
    run_args.kf_type = filter_type

    # Set a specific results folder for this run
    if filter_type == 'IEKF':
        run_name = f"{filter_type}_{iterations}iter"
    else:
        run_name = filter_type

    # Create specific folder for this run
    run_folder = os.path.join(args.results_folder, run_name)
    os.makedirs(run_folder, exist_ok=True)
    run_args.results_folder = run_folder

    print(f"\n{'='*50}")
    print(f"Running {run_name}...")
    print(f"{'='*50}")

    # Load map data
    map_data = blue_sky.Map().load(run_args, mock=False)

    # Create the true trajectory
    true_traj = blue_sky.CreateTraj(run_args).create(map_data)

    # Generate noisy trajectory
    errors = blue_sky.IMUErrors(run_args.imu_errors)
    errors.set_imu_error('position', amplitude=20, drift=0, bias=0)
    meas_traj = blue_sky.NoiseTraj(true_traj).add_noise(errors.imu_errors, approach='top-down')

    # Create and configure the filter
    if filter_type == 'IEKF':
        filter_instance = blue_sky.IEKF(run_args)
        # Set the number of iterations
        filter_instance.max_iter = iterations
    else:  # UKF
        filter_instance = blue_sky.UKF(run_args)

    # Run the filter
    start_time = time.time()
    estimation_results = filter_instance.run(map_data, meas_traj)
    run_time = time.time() - start_time

    # Calculate errors and covariances
    run_errors, covariances = blue_sky.calc_errors_covariances(true_traj, estimation_results)

    # Calculate CEP50
    north_errors = run_errors.pos.north
    east_errors = run_errors.pos.east
    radial_errors = np.sqrt(north_errors**2 + east_errors**2)
    cep50 = np.percentile(radial_errors, 50)
    cep90 = np.percentile(radial_errors, 90)

    # Create plots with comparison=False (we'll create comparison plots later)
    create_plots(run_args, map_data, true_traj, meas_traj, estimation_results,
                 run_errors, covariances, run_folder, comparison=False)

    # Return relevant results
    return {
        'name': run_name,
        'filter_instance': filter_instance,
        'estimation_results': estimation_results,
        'run_errors': run_errors,
        'covariances': covariances,
        'cep50': cep50,
        'cep90': cep90,
        'run_time': run_time
    }

def create_plots(args, map_data, true_traj, meas_traj, estimation_results,
                 run_errors, covariances, output_folder, comparison=False):
    """
    Create standard plots for a single filter run.

    Args:
        args: Configuration arguments
        map_data: Map data
        true_traj: True trajectory
        meas_traj: Measured trajectory
        estimation_results: Filter estimation results
        run_errors: Error metrics
        covariances: Covariance data
        output_folder: Where to save the plots
        comparison: Whether this is a comparison plot (affects styling)
    """
    # Define standard plot options
    plot_options = {
        'plot map': True,
        'position errors': True,
        'velocity errors': True,
        'attitude errors': True,
        'altitude errors': True,
        'model errors': True,
        'kalman gains': True,
        'map elevation': True,
    }

    # Apply plot options
    args.plots = plot_options

    # Generate plots
    blue_sky.plot_results(args, map_data, true_traj, meas_traj, estimation_results,
                          run_errors, covariances)

def create_comparison_plots(results, map_data, true_traj, meas_traj, output_folder):
    """
    Create comparison plots showing all filter performances together.

    Args:
        results: List of results from different filter runs
        map_data: Map data
        true_traj: True trajectory
        meas_traj: Measured trajectory
        output_folder: Where to save the comparison plots
    """
    # Create comparison plots directory
    comparison_dir = os.path.join(output_folder, 'comparison_plots')
    os.makedirs(comparison_dir, exist_ok=True)

    # 1. Position Error Comparison
    create_position_error_comparison(results, comparison_dir)

    # 2. Trajectory Comparison
    create_trajectory_comparison(results, true_traj, map_data, comparison_dir)

    # 3. CEP Comparison
    create_cep_comparison(results, comparison_dir)

    # 4. Error metrics comparison
    create_error_metrics_comparison(results, comparison_dir)

    # 5. Runtime comparison
    create_runtime_comparison(results, comparison_dir)

def create_position_error_comparison(results, output_dir):
    """Create position error comparison plot."""
    plt.figure(figsize=(12, 9))

    # North position errors
    plt.subplot(2, 1, 1)
    for result in results:
        plt.plot(result['run_errors'].pos.north,
                 label=f"{result['name']} (CEP50: {result['cep50']:.2f}m)")

    plt.title('North Position Error Comparison', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Error (m)')
    plt.grid(True)
    plt.legend()

    # East position errors
    plt.subplot(2, 1, 2)
    for result in results:
        plt.plot(result['run_errors'].pos.east,
                 label=f"{result['name']} (CEP50: {result['cep50']:.2f}m)")

    plt.title('East Position Error Comparison', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Error (m)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_error_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'position_error_comparison.svg'))
    plt.close()

def create_trajectory_comparison(results, true_traj, map_data, output_dir):
    """Create trajectory comparison plot."""
    plt.figure(figsize=(14, 10))

    # 2D trajectory comparison
    plt.subplot(2, 1, 1)

    # Plot map background
    X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
    plt.contourf(X, Y, map_data.grid, cmap='terrain', alpha=0.3)

    # Plot trajectories
    plt.plot(true_traj.pos.lon, true_traj.pos.lat, 'k-', linewidth=2, label='True Trajectory')

    for result in results:
        plt.plot(result['estimation_results'].traj.pos.lon,
                 result['estimation_results'].traj.pos.lat,
                 '-', linewidth=1.5, label=f"{result['name']}")

    plt.title('2D Trajectory Comparison', fontsize=14)
    plt.xlabel('Longitude (deg)')
    plt.ylabel('Latitude (deg)')
    plt.legend()
    plt.grid(True)

    # 3D trajectory comparison (position errors vs. time)
    plt.subplot(2, 1, 2)

    x = np.arange(len(true_traj.pos.lon))

    for result in results:
        # Calculate radial error
        north_error = result['run_errors'].pos.north
        east_error = result['run_errors'].pos.east
        radial_error = np.sqrt(north_error**2 + east_error**2)

        plt.plot(x, radial_error, '-', label=f"{result['name']} (CEP50: {result['cep50']:.2f}m)")

    plt.axhline(y=200, color='r', linestyle='--', label='200m threshold')

    plt.title('Radial Position Error Comparison', fontsize=14)
    plt.xlabel('Time Steps')
    plt.ylabel('Radial Error (m)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'trajectory_comparison.svg'))
    plt.close()

def create_cep_comparison(results, output_dir):
    """Create CEP comparison bar chart."""
    plt.figure(figsize=(10, 6))

    names = [result['name'] for result in results]
    cep50_values = [result['cep50'] for result in results]
    cep90_values = [result['cep90'] for result in results]

    x = np.arange(len(names))
    width = 0.35

    plt.bar(x - width/2, cep50_values, width, label='CEP50', color='steelblue')
    plt.bar(x + width/2, cep90_values, width, label='CEP90', color='lightcoral')

    # Add threshold line
    plt.axhline(y=200, color='r', linestyle='--', label='200m threshold')

    plt.xlabel('Filter Configuration')
    plt.ylabel('Error (meters)')
    plt.title('CEP Comparison Across Filter Configurations', fontsize=14)
    plt.xticks(x, names, rotation=45)

    # Add value labels on top of bars
    for i, v in enumerate(cep50_values):
        plt.text(i - width/2, v + 5, f"{v:.2f}", ha='center', fontsize=9)

    for i, v in enumerate(cep90_values):
        plt.text(i + width/2, v + 5, f"{v:.2f}", ha='center', fontsize=9)

    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'cep_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'cep_comparison.svg'))
    plt.close()

def create_error_metrics_comparison(results, output_dir):
    """Create error metrics comparison plot."""
    plt.figure(figsize=(12, 8))

    metrics = ['pos_north_rmse', 'pos_east_rmse', 'vel_north_rmse', 'vel_east_rmse', 'vel_down_rmse']
    metric_labels = ['North Position RMSE', 'East Position RMSE', 'North Velocity RMSE',
                     'East Velocity RMSE', 'Down Velocity RMSE']

    data = []

    for result in results:
        metrics_values = [
            result['run_errors'].metrics['pos']['north']['rmse'],
            result['run_errors'].metrics['pos']['east']['rmse'],
            result['run_errors'].metrics['vel']['north']['rmse'],
            result['run_errors'].metrics['vel']['east']['rmse'],
            result['run_errors'].metrics['vel']['down']['rmse']
        ]
        data.append(metrics_values)

    data = np.array(data).T  # Transpose for plotting

    x = np.arange(len(metric_labels))
    width = 0.8 / len(results)

    for i, result in enumerate(results):
        offset = (i - len(results)/2 + 0.5) * width
        bars = plt.bar(x + offset, data[:, i], width, label=result['name'])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f"{height:.2f}", ha='center', va='bottom', fontsize=8, rotation=90)

    plt.xlabel('Error Metric')
    plt.ylabel('RMSE Value')
    plt.title('Error Metrics Comparison Across Filter Configurations', fontsize=14)
    plt.xticks(x, metric_labels, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'error_metrics_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'error_metrics_comparison.svg'))
    plt.close()

def create_runtime_comparison(results, output_dir):
    """Create runtime comparison bar chart."""
    plt.figure(figsize=(10, 6))

    names = [result['name'] for result in results]
    runtimes = [result['run_time'] for result in results]

    plt.bar(names, runtimes, color='teal')

    plt.xlabel('Filter Configuration')
    plt.ylabel('Runtime (seconds)')
    plt.title('Computational Performance Comparison', fontsize=14)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, v in enumerate(runtimes):
        plt.text(i, v + 0.1, f"{v:.2f}s", ha='center')

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.svg'))
    plt.close()

def main():
    """
    Main function to run the filter comparison experiment.
    """
    # Setup experiment
    args, errors, results_dir = setup_experiment()

    # List to store all results
    all_results = []

    # Run IEKF with 1 iteration
    results_iekf_1 = run_filter(args, 'IEKF', iterations=1)
    all_results.append(results_iekf_1)

    # Run IEKF with 3 iterations
    results_iekf_3 = run_filter(args, 'IEKF', iterations=3)
    all_results.append(results_iekf_3)

    # Run IEKF with 5 iterations
    results_iekf_5 = run_filter(args, 'IEKF', iterations=5)
    all_results.append(results_iekf_5)

    # Run UKF
    results_ukf = run_filter(args, 'UKF')
    all_results.append(results_ukf)

    # First run to get map_data and trajectories for comparison plots
    map_data = blue_sky.Map().load(args, mock=False)
    true_traj = blue_sky.CreateTraj(args).create(map_data)
    meas_traj = blue_sky.NoiseTraj(true_traj).add_noise(errors.imu_errors, approach='top-down')

    # Create comparison plots
    create_comparison_plots(all_results, map_data, true_traj, meas_traj, results_dir)

    # Print summary table
    print("\n\n" + "="*80)
    print("FILTER COMPARISON SUMMARY".center(80))
    print("="*80)
    print(f"{'Filter':20} | {'CEP50 (m)':15} | {'CEP90 (m)':15} | {'Runtime (s)':15} | {'Result':10}")
    print("-"*80)

    for result in all_results:
        status = "PASSED" if result['cep50'] < 200 else "FAILED"
        print(f"{result['name']:20} | {result['cep50']:15.2f} | {result['cep90']:15.2f} | {result['run_time']:15.2f} | {status:10}")

    print("="*80)
    print(f"\nAll results and plots saved to: {results_dir}")
    print("\nFor presentations, check the 'comparison_plots' folder for comparative visualizations.")

if __name__ == "__main__":
    main()