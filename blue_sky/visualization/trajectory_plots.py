import matplotlib.pyplot as plt
import numpy as np
from .common import setup_figure, save_figure, compute_plot_limits


def plot_trajectory(traj, map_data, output_folder='out', save=False):
    """
    Plot the velocity components of a trajectory.

    Args:
        traj: Trajectory instance containing velocity data
        map_data: Map data for background
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Trajectory Visualization'
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    # 2D Plot as the first subplot
    ax1 = fig.add_subplot(211)
    X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
    ax1.contourf(X, Y, map_data.grid, cmap='terrain', alpha=0.5)
    ax1.plot(traj.pos.lon, traj.pos.lat, 'r-', label='2D Trajectory')
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')
    ax1.set_title('2D View of Trajectory on Map')
    ax1.legend()

    # 3D Plot as the second subplot
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot_surface(X, Y, map_data.grid, cmap='terrain', alpha=0.5)
    ax2.plot(traj.pos.lon, traj.pos.lat, traj.pos.h_asl, 'r-', label='3D Trajectory')
    ax2.set_xlabel('Longitude [deg]')
    ax2.set_ylabel('Latitude [deg]')
    ax2.set_zlabel('Altitude [m]')
    ax2.set_title('3D View of Trajectory on Map')
    ax2.legend()

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig

def plot_velocity(traj, output_folder='out', save=False):
    """
    Plot the velocity components of a trajectory.

    Args:
        traj: Trajectory instance containing velocity data
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Velocity Components'
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Plot North velocity component
    axs[0].plot(traj.time_vec, traj.vel.north, 'b-')
    axs[0].set_title('North Velocity')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Velocity [m/s]')
    axs[0].grid(True)

    # Plot East velocity component
    axs[1].plot(traj.time_vec, traj.vel.east, 'r-')
    axs[1].set_title('East Velocity')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Velocity [m/s]')
    axs[1].grid(True)

    # Plot Down velocity component
    axs[2].plot(traj.time_vec, traj.vel.down, 'g-')
    axs[2].set_title('Down Velocity')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Velocity [m/s]')
    axs[2].grid(True)

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_trajectory_views(traj, map_data, output_folder='out', save=False):
    """
    Plot the trajectory from North and East views.

    Args:
        traj: Trajectory instance
        map_data: Map data containing grid and axis information
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Trajectory Views'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # North View: Looking from North towards South
    # Here, we plot East vs. Altitude
    ax1.plot(traj.pos.east, traj.pos.h_asl, 'b-', label='View from North')
    ax1.set_xlabel('East [m]')
    ax1.set_ylabel('Altitude [m]')
    ax1.set_title('View from North')
    ax1.legend()
    ax1.grid(True)

    # East View: Looking from East towards West
    # Here, we plot North vs. Altitude
    ax2.plot(traj.pos.north, traj.pos.h_asl, 'g-', label='View from East')
    ax2.set_xlabel('North [m]')
    ax2.set_ylabel('Altitude [m]')
    ax2.set_title('View from East')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_trajectory_comparison(true_traj, map_data, amp=100, drift=0.5, bias=250, output_folder='out', save=False):
    """
    Plot a comparison between true trajectory and noisy trajectories.

    Args:
        true_traj: True trajectory instance
        map_data: Map data for background
        amp (float): Noise amplitude
        drift (float): Noise drift rate
        bias (float): Noise bias
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = 'Trajectory Comparison'
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    vec_shape = true_traj.pos.north.shape

    # Generate noise for top-down approach (only amplitude noise)
    top_down_north = true_traj.pos.north + amp * np.random.randn(*vec_shape)
    top_down_east = true_traj.pos.east + amp * np.random.randn(*vec_shape)

    # Generate noise for bottom-up approach (bias, drift, and amplitude noise)
    bottom_up_north = true_traj.pos.north + bias * np.ones(vec_shape)
    bottom_up_east = true_traj.pos.east + bias * np.zeros(vec_shape)

    bottom_up_north += np.cumsum(drift * np.ones(vec_shape), axis=0)
    bottom_up_east += np.cumsum(drift * np.random.randn(*vec_shape), axis=0)

    bottom_up_north += amp * np.random.randn(*vec_shape)
    bottom_up_east += amp * np.random.randn(*vec_shape)

    # Plot setup
    lat_grid, lon_grid = np.meshgrid(map_data.axis['lat'], map_data.axis['lon'], indexing='ij')
    north_grid = lat_grid * map_data.mpd['north']
    east_grid = lon_grid * map_data.mpd['east']

    # Plot map background and trajectories
    ax.contourf(east_grid, north_grid, map_data.grid, cmap='bone', alpha=0.6)
    ax.plot(top_down_east, top_down_north, 'b', linestyle=':', linewidth=2, label='Top-Down Trajectory')
    ax.plot(bottom_up_east, bottom_up_north, 'r', linestyle='-.', linewidth=2, label='Bottom-Up Trajectory')
    ax.plot(true_traj.pos.east, true_traj.pos.north, 'w', linestyle='-', linewidth=2, label='True Trajectory')

    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.legend()

    # Calculate limits with margin
    all_east = [true_traj.pos.east, top_down_east, bottom_up_east]
    all_north = [true_traj.pos.north, top_down_north, bottom_up_north]

    east_limits = compute_plot_limits(all_east, margin_percent=5)
    north_limits = compute_plot_limits(all_north, margin_percent=5)

    ax.set_xlim(east_limits)
    ax.set_ylim(north_limits)

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_pinpoint_trajectories_2d(true_traj, map_data, output_folder='out', save=False):
    """
    Plot the true trajectory, measured trajectory, and pinpoint results in 2D.

    Args:
        true_traj: True trajectory data
        map_data: Map data containing grid and axis information
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = "PinPoint Results in 2D"
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Sample index for points
    idx = 50

    # Plot map background
    X, Y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
    ax.contourf(X, Y, map_data.grid, cmap='bone', alpha=0.5)

    # Plot true trajectory
    ax.plot(true_traj.pos.lon, true_traj.pos.lat, 'k--', label='True Trajectory')

    # Plot measured trajectory
    measured_lon = true_traj.pos.lon[::idx]
    measured_lat = true_traj.pos.lat[::idx]
    ax.scatter(measured_lon, measured_lat, color='blue', marker='o', label='Measured Trajectory')

    # Plot pinpoint results
    pinpoint_lon = true_traj.pinpoint.lon[::idx]
    pinpoint_lat = true_traj.pinpoint.lat[::idx]
    ax.scatter(pinpoint_lon, pinpoint_lat, color='red', marker='x', label='Pinpoint Results')

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.legend()

    # Set appropriate limits
    lon_limits = compute_plot_limits([pinpoint_lon], margin_percent=5)
    lat_limits = compute_plot_limits([pinpoint_lat], margin_percent=5)

    ax.set_xlim(lon_limits)
    ax.set_ylim(lat_limits)

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig


def plot_pinpoint_trajectories_3d(true_traj, map_data, pov=(50, 200), zoom=2.3, output_folder='out', save=False):
    """
    Plot the true trajectory, measured trajectory, and pinpoint results in 3D.

    Args:
        true_traj: True trajectory data
        map_data: Map data containing grid and axis information
        pov (tuple): Point of view (elevation, azimuth)
        zoom (float): Zoom factor
        output_folder (str): Folder to save the figure in
        save (bool): Whether to save the figure

    Returns:
        matplotlib.figure.Figure: The figure
    """
    title = "PinPoint Results in 3D"
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle(title, fontsize=24, fontweight='bold')

    # Sample index for points
    idx = 50

    # Create 3D axis
    ax = fig.add_subplot(111, projection='3d')

    # Plot true trajectory
    ax.plot(true_traj.pos.lon, true_traj.pos.lat, true_traj.pos.h_asl, 'k--', label='True Trajectory')

    # Plot measured trajectory
    measured_lon = true_traj.pos.lon[::idx]
    measured_lat = true_traj.pos.lat[::idx]

    # Get ground elevations at measured points
    ground_elevations_measured = []
    for lat, lon in zip(measured_lat, measured_lon):
        lat_idx = np.abs(map_data.axis['lat'] - lat).argmin()
        lon_idx = np.abs(map_data.axis['lon'] - lon).argmin()
        ground_elevations_measured.append(map_data.grid[lat_idx, lon_idx])

    ax.scatter(measured_lon, measured_lat, ground_elevations_measured,
               color='blue', marker='o', label='Measured Trajectory')

    # Plot pinpoint results
    pinpoint_lon = true_traj.pinpoint.lon[::idx]
    pinpoint_lat = true_traj.pinpoint.lat[::idx]

    # Get ground elevations at pinpoint locations
    ground_elevations_pinpoint = []
    for lat, lon in zip(pinpoint_lat, pinpoint_lon):
        lat_idx = np.abs(map_data.axis['lat'] - lat).argmin()
        lon_idx = np.abs(map_data.axis['lon'] - lon).argmin()
        ground_elevations_pinpoint.append(map_data.grid[lat_idx, lon_idx])

    ax.scatter(pinpoint_lon, pinpoint_lat, ground_elevations_pinpoint,
               color='red', marker='x', label='Pinpoint Results')

    ax.set_xlabel('Longitude [deg]')
    ax.set_ylabel('Latitude [deg]')
    ax.set_zlabel('Altitude [m]')
    ax.legend()

    # Set the view angle
    ax.view_init(elev=pov[0], azim=pov[1])

    # Apply zoom
    x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
    z_mid = (ax.get_zlim()[0] + ax.get_zlim()[1]) / 2

    x_range = (ax.get_xlim()[1] - ax.get_xlim()[0]) / zoom
    y_range = (ax.get_ylim()[1] - ax.get_ylim()[0]) / zoom
    z_range = (ax.get_zlim()[1] - ax.get_zlim()[0]) / zoom

    ax.set_xlim([x_mid - x_range / 2, x_mid + x_range / 2])
    ax.set_ylim([y_mid - y_range / 2, y_mid + y_range / 2])
    ax.set_zlim([z_mid - z_range / 2, z_mid + z_range / 2])

    plt.tight_layout()

    if save:
        save_figure(fig, title, output_folder)

    return fig