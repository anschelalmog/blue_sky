import numpy as np
from scipy.interpolate import RegularGridInterpolator
from modules.utils import cosd, sind
from modules.base_traj import BaseTraj
from modules.pinpoint_calc import PinPoint


class CreateTraj(BaseTraj):
    def __init__(self, args):
        super().__init__(args.run_points)
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        #
        self.init_lat = args.init_lat
        self.init_lon = args.init_lon
        self.init_height = args.init_height
        #
        self.avg_spd = args.avg_spd
        #
        self.init_psi = args.psi
        self.init_theta = args.theta
        self.init_phi = args.phi
        #
        self.pinpoint = None

    def create_linear(self, map_data):
        # assuming constant velocity, straight flight

        self._create_euler_linear()
        self._create_vel_linear()
        self._create_pos_linear(map_data)
        self._create_traj_linear(map_data)
        self.pinpoint = PinPoint(self.run_points).calc(self, map_data)
        return self

    def _create_euler_linear(self):
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
        self.euler.psi = self.init_psi * np.ones(self.run_points)  # [deg]
        self.euler.theta = self.init_theta * np.ones(self.run_points)  # [deg]
        self.euler.phi = self.init_phi * np.ones(self.run_points)  # [deg]

    def _create_vel_linear(self):
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
        self.vel.north = self.avg_spd * cosd(self.init_psi) * np.ones(self.run_points)  # [m/s]
        self.vel.east = self.avg_spd * sind(self.init_psi) * np.ones(self.run_points)  # [m/s]
        self.vel.down = self.avg_spd * sind(self.init_theta) * np.ones(self.run_points)  # [m/s]

    def _create_pos_linear(self, map_data):
        """
        creating position vectors
        assuming: straight flight, constant velocity
        """
        final_lat = map_data.final_pos[0]
        final_lon = map_data.final_pos[1]
        self.pos.h_asl = self.init_height * np.ones(self.run_points)

        # 501 arbitrary to avoid interpolation problems later
        x = 501
        self.pos.north = map_data.ax_north[x] + self.vel.north * self.time_vec
        self.pos.east = map_data.ax_east[x] + self.vel.east * self.time_vec

        self.pos.lat = np.linspace(self.init_lat, final_lat, self.pos.north.size)
        self.pos.lon = np.linspace(self.init_lon, final_lon, self.pos.east.size)

        self.mpd_north = self.pos.north / self.pos.lat  # [m/deg]
        self.mpd_east = self.pos.east / self.pos.lon  # [m/deg]

    def _create_traj_linear(self, map_data):
        """
              Interpolate a map grid at the trajectory points and calculate the corresponding trajectory heights.
             param:
                 map_data (MapData): An instance of a class that contains the map grid data
             return:
                 h_map -  A 2D array of the interpolated map grid at the latitude and longitude points
                         corresponding to the trajectory
                 traj_heights - A 1D array of interpolated heights at the trajectory's specific latitude
                                and longitude points
             """
        interpolator = RegularGridInterpolator((map_data.ax_lat, map_data.ax_lon), map_data.grid)
        points = np.array(np.meshgrid(self.pos.lat, self.pos.lon)).T.reshape(-1, 2)
        self.pos.h_map_grid = interpolator(points).reshape(self.pos.lat.size, self.pos.lon.size)
        self.pos.h_map = interpolator(np.vstack((self.pos.lat, self.pos.lon)).T)
