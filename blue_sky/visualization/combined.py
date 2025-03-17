import matplotlib
import matplotlib.pyplot as plt

from .trajectory_plots import plot_trajectory
from .error_plots import (
    plot_position_errors,
    plot_velocity_errors,
    plot_altitude_errors,
    plot_attitude_errors,
    plot_model_errors,
    plot_kalman_gains,
    plot_map_elevation
)


def plot_results(args, map_data, ground_truth, measurements, estimation_results, errors, covars):
    """
    Create and display all visualization plots based on the plot options.

    Args:
        args: Configuration arguments
        map_data: Map data for terrain visualization
        ground_truth: True trajectory data
        measurements: Measured trajectory data
        estimation_results: Results from estimation algorithm
        errors: Error data between true and estimated trajectories
        covars: Covariance data for error bounds

    Returns:
        tuple: (errors, covars) to maintain compatibility with original code
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs('out', exist_ok=True)

    # Generate plots based on options
    if args.plots.get('plot map', False):
        plot_trajectory(ground_truth, map_data, save=True, output_folder=args.results_folder)
        #plt.show()

    if args.plots.get('position errors', False):
        # Use meas_traj.pos.north and meas_traj.pos.lon for compatibility
        errors.pos.north = estimation_results.traj.pos.north
        errors.pos.east = estimation_results.traj.pos.lon
        plot_position_errors(args, errors, covars, save=True, output_folder=args.results_folder, sample_size=25)
        plt.show()

    if args.plots.get('velocity errors', False):

        plot_velocity_errors(args, errors, covars, save=True, output_folder=args.results_folder)
        plt.show()

    if args.plots.get('attitude errors', False):
        plot_attitude_errors(args, errors, covars, save=True, output_folder=args.results_folder)
        plt.show()
    if args.plots.get('altitude errors', False):
        plot_altitude_errors(args, errors, covars, estimation_results, save=True, output_folder=args.results_folder)
        plt.show()
    if args.plots.get('model errors', False):
        plot_model_errors(args, estimation_results, save=True, output_folder=args.results_folder)
        plt.show()
    if args.plots.get('kalman gains', False):
        plot_kalman_gains(args, estimation_results, save=True, output_folder=args.results_folder)
        plt.show()
    if args.plots.get('map elevation', False):
        plot_map_elevation(args, ground_truth, estimation_results, save=True, output_folder=args.results_folder)
        plt.show()

    return errors, covars
