import matplotlib.pyplot as plt
import numpy as np
from .common import setup_figure, save_figure, create_error_plot


def plot_position_errors(args, errors, covars, output_folder='out', save=False, sample_size=None):
    """
    Plot position errors with covariance bounds.

    Args:
        args: Configuration arguments with time vector
        errors: Error data for position components
        covars: Covariance data for position components
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure
        sample_size (int): Number of points to sample (None for all)

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Position Errors'
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # If sample_size is provided, sample the data
    if sample_size is not None and sample_size < len(args.time_vec):
        time_vec = args.time_vec[:sample_size]
        north_errors = errors.pos.north[:sample_size]
        east_errors = errors.pos.east[:sample_size]
        north_covars = covars.pos.north[:sample_size]
        east_covars = covars.pos.east[:sample_size]
    else:
        time_vec = args.time_vec
        north_errors = errors.pos.north
        east_errors = errors.pos.east
        north_covars = covars.pos.north
        east_covars = covars.pos.east

    # Error in North position
    create_error_plot(time_vec, north_errors, north_covars,
                      'North Position Error [m]', 'Error [m]', axs[0])

    # Error in East Position
    create_error_plot(time_vec, east_errors, east_covars,
                      'East Position Error [m]', 'Error [m]', axs[1])

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_velocity_errors(args, errors, covars, output_folder='out', save=False):
    """
    Plot velocity errors with covariance bounds.

    Args:
        args: Configuration arguments with time vector
        errors: Error data for velocity components
        covars: Covariance data for velocity components
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Position Errors'
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Error in North Velocity
    copy_north_err = errors.vel.north
    copy_north_covars = covars.vel.north
    create_error_plot(args.time_vec,copy_north_err, copy_north_covars,
                      'North Position Error [m]', 'Error [m]', axs[0])

    # Error in East Velocity
    copy_east_err = errors.vel.east
    copy_east_covars = covars.vel.east
    create_error_plot(args.time_vec, copy_east_err, copy_east_covars,
                      'East Position Error [m]', 'Error [m]', axs[1])

    # Error in Down Velocity
    copy_down_err = errors.vel.down
    copy_down_covars = covars.vel.down
    create_error_plot(args.time_vec, copy_down_err, copy_down_covars,
                      'Vertical Error [m]', 'Error [m]', axs[2])

    plt.tight_layout()
    plt.show()


    if save:
        save_figure(fig, title, output_folder)
    plt.show()
    return fig


def plot_altitude_errors(args, errors, covars, estimation_results, output_folder='out', save=False):
    """
    Plot altitude errors with covariance bounds.

    Args:
        args: Configuration arguments with time vector
        errors: Error data for altitude components
        covars: Covariance data for altitude components
        estimation_results: Results from estimation algorithm
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Altitude Errors'
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Plot error and covariance bounds
    ax.plot(args.time_vec, errors.pos.h_asl, 'r', linewidth=1)
    ax.plot(args.time_vec, covars.pos.h_asl, '--b', linewidth=1)
    ax.plot(args.time_vec, -covars.pos.h_asl, '--b', linewidth=1)

    # Plot mean error
    mean_err_alt = np.mean(errors.pos.h_asl)
    ax.plot(args.time_vec, mean_err_alt * np.ones(len(errors.pos.h_asl)), '*-k')

    # Plot innovation sequence
    ax.plot(args.time_vec, estimation_results.params.Z, '-.g')

    # Set labels and title
    ax.set_title('Altitude Error [m]')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Error [m]')
    ax.grid(True)

    # Add legend
    ax.legend(['Error', r'$\pm\sigma$', 'Mean Error', 'Innovation'], loc='upper right')

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_attitude_errors(args, errors, covars, output_folder='out', save=False):
    """
    Plot attitude (Euler angle) errors with covariance bounds.

    Args:
        args: Configuration arguments with time vector
        errors: Error data for attitude components
        covars: Covariance data for attitude components
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Attitude Errors'
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Error in euler psi (yaw)
    create_error_plot(args.time_vec, errors.euler.psi, covars.euler.psi,
                      r'Euler $\psi$ - yaw Error [deg]', 'Error [deg]', axs[0])

    # Error in euler theta (pitch)
    create_error_plot(args.time_vec, errors.euler.theta, covars.euler.theta,
                      r'Euler $\theta$ - pitch Error [deg]', 'Error [deg]', axs[1])

    # Error in euler phi (roll)
    create_error_plot(args.time_vec, errors.euler.phi, covars.euler.phi,
                      r'Euler $\phi$ - roll Error [deg]', 'Error [deg]', axs[2])

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_model_errors(args, estimation_results, output_folder='out', save=False):
    """
    Plot model-related errors.

    Args:
        args: Configuration arguments with time vector
        estimation_results: Results from estimation algorithm
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Model Errors'
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Measurement mismatch and error correction
    axs[0].set_title('Measurement Mismatch and Error Correction')
    axs[0].plot(args.time_vec, estimation_results.params.Z, '--b')
    axs[0].plot(args.time_vec, estimation_results.params.dX[2, :], '-r')
    axs[0].set_ylabel('Error [m]')
    axs[0].set_xlabel('Time [sec]')
    axs[0].grid(True)
    axs[0].legend(['Mismatch', 'Error Correction'], loc='best')

    # Process noise components
    axs[1].set_title('Process Noise Components')
    axs[1].plot(args.time_vec, estimation_results.params.Rc, '-r')
    axs[1].plot(args.time_vec, estimation_results.params.Rfit, '--b')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_xlabel('Time [sec]')
    axs[1].grid(True)
    axs[1].legend(['Rc - Height Penalty', 'Rfit - Fit Error'], loc='best')

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_kalman_gains(args, estimation_results, output_folder='out', save=False):
    """
    Plot Kalman gain values.

    Args:
        args: Configuration arguments with time vector
        estimation_results: Results from estimation algorithm
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = "Kalman Gains"
    fig, axs = plt.subplots(2, 3, figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Define gain titles
    gains = [
        'Position North',
        'Position East',
        'Position Down',
        'Velocity North',
        'Velocity East',
        'Velocity Down'
    ]

    # Plot each gain
    for i, gain_title in enumerate(gains):
        row = i // 3
        col = i % 3

        axs[row, col].set_ylim(-1, 1.1)
        axs[row, col].grid(True)
        axs[row, col].plot(args.time_vec, estimation_results.params.K[i, :], linewidth=1)
        axs[row, col].set_title(gain_title)
        axs[row, col].set_xlabel('Time [sec]')

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_map_elevation(args, ground_truth, estimation_results, output_folder='out', save=False):
    """
    Plot map elevation comparison between true and estimated trajectories.

    Args:
        args: Configuration arguments with time vector
        ground_truth: True trajectory data
        estimation_results: Results from estimation algorithm
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Map Elevation'
    fig = plt.figure(figsize=(10, 12))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    plt.plot(args.time_vec, ground_truth.pinpoint.h_map, 'r')
    plt.plot(args.time_vec, estimation_results.traj.pos.h_asl - estimation_results.traj.pos.h_agl, '--b')

    plt.title('Map Elevation at Trajectory Points')
    plt.ylabel('Height [m]')
    plt.xlabel('Time [sec]')
    plt.grid(True)
    plt.legend(['True', 'Estimated'])

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig