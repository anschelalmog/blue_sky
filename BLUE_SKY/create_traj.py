import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from BLUE_SKY.utils import cosd, sind, get_mpd, DCM
from BLUE_SKY.base_traj import BaseTraj
from BLUE_SKY.pinpoint_calc import PinPoint
from BLUE_SKY.decorators import handle_interpolation_error


class CreateTraj(BaseTraj):
    """
    A class to create and handle trajectory data.

    Attributes:
    -----------
    run_points : int: Number of points in the trajectory.
    time_vec : numpy.ndarray: Array of time values corresponding to each point in the trajectory.
    inits : dict: Initial conditions and parameters for the trajectory calculation.
    pinpoint : PinPoint: An instance of PinPoint for pinpoint calculations.

    Methods:
    --------
    _create_euler(): Creates Euler angle vectors.
    _create_acc(): Creates acceleration vectors.
    _create_vel(): Creates velocity vectors.
    _create_pos(): Creates position vectors considering constant acceleration.
    _create_traj(map_data): Interpolates map data at trajectory points to calculate corresponding heights.
    create(map_data): Creates the full trajectory including Euler angles, acceleration, velocity, and position.
    plot_vel(): Plots the velocity components (North, East, Down) as functions of time in separate subplots.
    plot_trajectory(map_data): Plots the trajectory on the map in both 2D and 3D as subplots of the same figure.
    plot_views(map_data): Plots the trajectory from North and East views.
    """
    def __init__(self, args):
        super().__init__(args.run_points)
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        self.inits = {'lat': args.init_lat, 'lon': args.init_lon, 'height': args.init_height,
                      'avg_spd': args.avg_spd, 'acc_north': args.acc_north, 'acc_east': args.acc_east,
                      'acc_down': args.acc_down, 'psi': args.psi, 'theta': args.theta, 'phi': args.phi,
                      'psi_dot': args.psi_dot, 'theta_dot': args.theta_dot, 'phi_dot': args.phi_dot}
        self.pinpoint = None

    def create(self, map_data):
        self._create_euler()
        self._create_acc()
        self._create_vel()
        self._create_pos()
        self._apply_dcm(map_data)
        self._create_traj(map_data)
        self.pinpoint = PinPoint(self.run_points).calc(self, map_data)
        self.pos.h_agl = self.pos.h_asl - self.pinpoint.h_map
        return self

    def _create_euler(self):
        """
        creating euler angels vectors
        assuming the angels change linearly
        :returns euler angels vectors in [deg]
        """
        self.euler.psi = self.inits['psi'] + self.inits['psi_dot'] * self.time_vec
        self.euler.theta = self.inits['theta'] + self.inits['theta_dot'] * self.time_vec
        self.euler.phi = self.inits['phi'] + self.inits['phi_dot'] * self.time_vec

    def _create_acc(self):
        """
        Creates the acceleration vectors
        """
        self.acc.north = self.inits['acc_north'] * np.ones(self.run_points)
        self.acc.east = self.inits['acc_east'] * np.ones(self.run_points)
        self.acc.down = self.inits['acc_down'] * np.ones(self.run_points)

    def _create_vel(self):
        """
        creating velocity vectors

                # North axis
                ^
                |       /
            cos |  phi /
                |     /
                |    /
                |   /
                |  /
                +----------------> East axis
                    sin
        :returns velocity vectors in [m/s]
        """
        self.vel.north = self.inits['avg_spd'] * cosd(self.inits['psi']) + self.acc.north * self.time_vec
        self.vel.east = self.inits['avg_spd'] * sind(self.inits['psi']) + self.acc.east * self.time_vec
        self.vel.down = self.inits['avg_spd'] * sind(self.inits['theta']) + self.acc.down * self.time_vec

    def _create_pos(self):
        """
        Update position vectors considering constant acceleration.

        """
        self.mpd_north, self.mpd_east = get_mpd(self.inits['lat'])
        init_north, init_east = self.inits['lat'] * self.mpd_north, self.inits['lon'] * self.mpd_east

        # X = X_0 + V_0 * t  + 0.5 * a * (t^2)
        self.pos.north = (init_north + self.vel.north[0] * self.time_vec +
                          0.5 * self.inits['acc_north'] * self.time_vec ** 2)
        self.pos.east = (init_east + self.vel.east[0] * self.time_vec +
                         0.5 * self.inits['acc_east'] * self.time_vec ** 2)

        self.pos.h_asl = self.inits['height'] + self.vel.down[0] - 0.5 * self.inits['acc_down'] * self.time_vec ** 2

        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east

        self.mpd_north, self.mpd_east = get_mpd(self.pos.lat)

        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east

    def _apply_dcm(self, map_data):
        """
        Apply the Direction Cosine Matrix (DCM) to the position offset of the IMU sensor at each time step.
        """
        transformed_offsets = np.zeros((self.run_points, 3))

        for i in range(self.run_points):
            # Update IMU offset with the current height above sea level
            imu_offset = np.array([0, 0, -self.pos.h_asl[i]])

            dcm = DCM(yaw=self.euler.psi[i], pitch=self.euler.theta[i], roll=self.euler.phi[i])

            # Apply the DCM transformation to the IMU offset
            transformed_offset = dcm.matrix @ imu_offset
            transformed_offsets[i, :] = transformed_offset

            # Update latitude and longitude based on transformed offsets
            self.pos.lat[i] += transformed_offset[0] / self.mpd_north[i]
            self.pos.lon[i] += transformed_offset[1] / self.mpd_east[i]

        # Update measured height map based on the new lat and lon
        self._update_measured_height_map(map_data)

        return transformed_offsets

    def _update_measured_height_map(self, map_data):
        """
        Interpolates the map data at the new trajectory points to calculate the corresponding heights.
        """
        interpolator = RegularGridInterpolator((map_data.axis['lat'], map_data.axis['lon']),
                                               map_data.grid)
        points = np.vstack((self.pos.lat, self.pos.lon)).T
        self.pos.h_map = interpolator(points)

    @handle_interpolation_error
    def _create_traj(self, map_data):
        """
        Interpolate map data at trajectory points to calculate corresponding heights.

        :param map_data: An instance containing the map grid data.
        """
        interpolator = RegularGridInterpolator((map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
        points = np.vstack((self.pos.lat, self.pos.lon)).T
        self.pos.h_map = interpolator(points)

    def plot_vel(self):
        """
        Plots the velocity components (North, East, Down) as functions of time in separate subplots.
        """
        fig, axs = plt.subplots(3, 1,
                                figsize=(10, 15))  # Create a figure and a set of subplots with 3 rows and 1 column

        # Plot North velocity component
        axs[0].plot(self.time_vec, self.vel.north, 'b-')
        axs[0].set_title('North Velocity')
        axs[0].set_xlabel('Time [s]')
        axs[0].set_ylabel('Velocity [m/s]')
        axs[0].grid(True)

        # Plot East velocity component
        axs[1].plot(self.time_vec, self.vel.east, 'r-')
        axs[1].set_title('East Velocity')
        axs[1].set_xlabel('Time [s]')
        axs[1].set_ylabel('Velocity [m/s]')
        axs[1].grid(True)

        # Plot Down velocity component
        axs[2].plot(self.time_vec, self.vel.down, 'g-')
        axs[2].set_title('Down Velocity')
        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Velocity [m/s]')
        axs[2].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def plot_trajectory(self, map_data):
        """
        Plots the trajectory on the map in both 2D and 3D as subplots of the same figure.

        :param map_data: The map data containing the grid and axis information.
        """
        fig = plt.figure(figsize=(16, 8))

        # 2D Plot as the first subplot
        ax1 = fig.add_subplot(211)
        X, Y = np.meshgrid(map_data.ax_lon, map_data.ax_lat)
        ax1.contourf(X, Y, map_data.grid, cmap='terrain', alpha=0.5)
        ax1.plot(self.pos.lon, self.pos.lat, 'r-', label='2D Trajectory')
        ax1.set_xlabel('Longitude [deg]')
        ax1.set_ylabel('Latitude [deg]')
        ax1.set_title('2D View of Trajectory on Map')
        ax1.legend()

        # 3D Plot as the second subplot
        ax2 = fig.add_subplot(212, projection='3d')
        ax2.plot_surface(X, Y, map_data.grid, cmap='terrain', alpha=0.5)
        ax2.plot(self.pos.lon, self.pos.lat, self.pos.h_asl, 'r-', label='3D Trajectory')
        ax2.set_xlabel('Longitude [deg]')
        ax2.set_ylabel('Latitude [deg]')
        ax2.set_zlabel('Altitude [m]')
        ax2.set_title('3D View of Trajectory on Map')
        ax2.legend()

        # Set the overall figure title
        title = 'Trajectory Visualization'
        fig.suptitle(title, fontsize=16)
        plt.savefig(f'{title}.png')
        plt.show()

    def plot_views(self, map_data):
        """
        Plots the trajectory from North and East views.

        :param map_data: The map data containing the grid and axis information.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # North View: Looking from North towards South
        # Here, we plot East vs. Altitude
        ax1.plot(self.pos.east, self.pos.h_asl, 'b-', label='View from North')
        ax1.set_xlabel('East [m]')
        ax1.set_ylabel('Altitude [m]')
        ax1.set_title('View from North')
        ax1.legend()
        ax1.grid(True)

        # East View: Looking from East towards West
        # Here, we plot North vs. Altitude
        ax2.plot(self.pos.north, self.pos.h_asl, 'g-', label='View from East')
        ax2.set_xlabel('North [m]')
        ax2.set_ylabel('Altitude [m]')
        ax2.set_title('View from East')
        ax2.legend()
        ax2.grid(True)

        # Set the overall figure title
        title = 'Trajectory Views'
        fig.suptitle(title, fontsize=16)
        plt.savefig(f'{title}.png')

        plt.show()
