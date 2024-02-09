import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
from src.utils import cosd, sind, get_mpd
from src.base_traj import BaseTraj
from src.pinpoint_calc import PinPoint


class CreateTraj(BaseTraj):
    def __init__(self, args):
        super().__init__(args.run_points)
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        self.inits = {'lat': args.init_lat, 'lon': args.init_lon, 'height': args.init_height,
                    'avg_spd': args.avg_spd, 'acc_north': args.acc_north, 'acc_east': args.acc_east, 'acc_down': args.acc_down,
                    'psi_dot': args.psi_dot, 'theta_dot': args.theta_dot, 'phi_dot': args.phi_dot}
        self.pinpoint = None

    def _create_euler(self):
        """
        creating euler angels vectors

                          Z
                          ^
                          |
                          |
                          |     /|  Roll (φ)
                          |    / |
                          |   /  |
                          |  /   |
                          | /    |
                          |/     |
                          +------|---------> Y
                         / \     |
                        /   \    |
                       /     \   | Pitch (θ)
                      /       \  |
                     /         \ |
                    /           \|
                    X                 v Yaw (ψ)
        """
        self.euler.psi = self.inits['psi'] + self.psi_dot * self.time_vec  # [deg]
        self.euler.theta = self.inits['theta'] + self.theta_dot * self.time_vec  # [deg]
        self.euler.phi = self.inits.phi['phi'] + self.psi_dot * self.time_vec  # [deg]

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
        """
        self.vel.north = self.avg_spd * cosd(self.init_psi) + self.acc_north * self.time_vec  # [m/s]
        self.vel.east = self.avg_spd * sind(self.init_psi) + self.acc_north * self.time_vec  # [m/s]
        # self.vel.down = self.avg_spd * sind(self.init_phi) + self.acc_down * self.time_vec  # [m/s]
        self.vel.down += self.acc_down * self.time_vec  # [m/s]

    def _create_pos(self, map_data):
        """
        Update position vectors considering constant acceleration.
        """
        # self.mpd_north, self.mpd_east = get_mpd(self.init_lat)
        # init_north, init_east = self.init_lat / self.mpd_north, self.init_lon / self.mpd_east
        # # X = X_0 + V_0 * t  + 0.5 * a * (t^2)
        # self.pos.north = self.init_lat * self.mpd_north + self.vel.north[0] * self.time_vec + 0.5 * self.acc_north * self.time_vec ** 2
        # self.pos.east = self.init_lon * self.mpd_east + self.vel.east[0] * self.time_vec + 0.5 * self.acc_east * self.time_vec ** 2
        # self.pos.h_asl = self.init_height + self.vel.down[0] - 0.5 * self.acc_down * self.time_vec ** 2
        #
        # # Instead of using a single initial velocity value, use the entire velocity vector
        # self.pos.north = self.init_lat * self.mpd_north + np.cumsum(self.vel.north[:-1] + self.vel.north[1:]) / 2 * np.diff(self.time_vec)
        # self.pos.east = self.init_lon * self.mpd_east + np.cumsum(self.vel.east[:-1] + self.vel.east[1:]) / 2 * np.diff(self.time_vec)

        # For downward velocity, considering it starts from a constant value and changes due to acceleration
        self.pos.h_asl = self.init_height + np.cumsum(self.vel.down[:-1] + self.vel.down[1:]) / 2 * np.diff(self.time_vec)

        # Integrate velocity to get displacement, assuming linear motion
        # Displacement = velocity * time for each component
        displacement_north = np.cumsum(self.vel_north[:-1] * np.diff(self.time_vec))
        displacement_east = np.cumsum(self.vel_east[:-1] * np.diff(self.time_vec))
        displacement_down = np.cumsum(self.vel_down[:-1] * np.diff(self.time_vec))

        # Initial displacement is zero
        displacement_north = np.insert(displacement_north, 0, 0)
        displacement_east = np.insert(displacement_east, 0, 0)
        displacement_down = np.insert(displacement_down, 0, 0)

        # Update position by adding displacement to initial position
        self.pos_north = init_north + displacement_north
        self.pos_east = init_east + displacement_east
        self.pos_h_asl = self.init_height - displacement_down

        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east

        self.mpd_north, self.mpd_east = get_mpd(self.pos.lat)

    def _create_traj(self, map_data):
        """
        Interpolate map data at trajectory points to calculate corresponding heights.
        Uses updated latitudes and longitudes from _create_pos_linear function.

        :param map_data: An instance containing the map grid data.
        """
        interpolator = RegularGridInterpolator((map_data.ax_lat, map_data.ax_lon), map_data.grid)
        points = np.vstack((self.pos.lat, self.pos.lon)).T
        self.pos.h_map = interpolator(points)

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

    def create(self, map_data):
        self._create_euler()
        self._create_vel()
        self._create_pos(map_data)
        self._create_traj(map_data)
        self.pinpoint = PinPoint(self.run_points).calc(self, map_data)
        self.pos.h_agl = self.pos.h_asl - self.pinpoint.h_map
        return self
