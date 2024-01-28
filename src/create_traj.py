import numpy as np
from scipy.interpolate import RegularGridInterpolator
from src.utils import cosd, sind, get_mpd
from src.base_traj import BaseTraj
from src.pinpoint_calc import PinPoint


class CreateTraj(BaseTraj):
    def __init__(self, args):
        super().__init__(args.run_points)
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        # position
        self.init_lat = args.init_lat
        self.init_lon = args.init_lon
        self.init_height = args.init_height
        # velocity
        self.avg_spd = args.avg_spd
        self.acc_north = args.acc_north
        self.acc_east = args.acc_east
        self.acc_down = args.acc_down
        # euler angels
        self.init_psi = args.psi
        self.init_theta = args.theta
        self.init_phi = args.phi
        self.psi_dot = args.psi_dot
        self.theta_dot = args.theta_dot
        self.phi_dot = args.phi_dot

        self.pinpoint = None

    def _create_euler(self):
        """
            creating euler angels vectors
            assuming: straight flight

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
        self.euler.psi = self.init_psi * self.psi_dot * self.time_vec  # [deg]
        self.euler.theta = self.init_theta + self.theta_dot * self.time_vec  # [deg]
        self.euler.phi = self.init_phi + self.psi_dot * self.time_vec  # [deg]

    def _create_vel(self):
        """
           creating velocity vectors
           assuming: straight flight, constant velocity

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
        # assuming constant acceleration over small differences
        self.vel.north = self.avg_spd * cosd(self.init_psi) + self.acc_north * self.time_vec  # [m/s]
        self.vel.east = self.avg_spd * sind(self.init_psi) + self.acc_north * self.time_vec  # [m/s]
        self.vel.down = self.vel.down[0] + self.acc_north * self.time_vec  # [m/s]

    def _create_pos(self, map_data):
        """
        Update position vectors considering constant acceleration.
        """
        self.mpd_north, self.mpd_east = get_mpd(self.init_lat)
        # X = X_0 + V_0 * t  + 0.5 * a * (t^2)
        self.pos.north = self.vel.north[0] * self.time_vec + 0.5 * self.acc_north * self.time_vec ** 2
        self.pos.east = self.vel.east[0] * self.time_vec + 0.5 * self.acc_east * self.time_vec ** 2
        self.pos.h_asl = self.init_height - 0.5 * self.acc_down * self.time_vec ** 2
        self.pos.lat = self.init_lat + self.pos.north / self.mpd_north
        self.pos.lon = self.init_lon + self.pos.east / self.mpd_east

        self.mpd_north = self.pos.north / self.pos.lat
        self.mpd_east = self.pos.east / self.pos.lon

    def _create_traj(self, map_data):
        """
        Interpolate map data at trajectory points to calculate corresponding heights.
        Uses updated latitudes and longitudes from _create_pos_linear function.

        :param map_data: An instance containing the map grid data.
        """
        interpolator = RegularGridInterpolator((map_data.ax_lat, map_data.ax_lon), map_data.grid)
        points = np.vstack((self.pos.lat, self.pos.lon)).T
        self.pos.h_map = interpolator(points)

    def create(self, map_data):
        self._create_euler()
        self._create_vel()
        self._create_pos(map_data)
        self._create_traj(map_data)
        self.pinpoint = PinPoint(self.run_points).calc(self, map_data)
        return self