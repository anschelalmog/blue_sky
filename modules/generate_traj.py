from .base_classes import BaseTraj
from .pinpoint import PinPointCalc
import numpy as np
import numpy.random as rnd
from scipy.interpolate import RegularGridInterpolator
from .utils import cosd, sind
import matplotlib.pyplot as plt


class CreateTraj(BaseTraj):
    def __init__(self, args, map_data):
        super().__init__(args)
        self.time_vec = args.time_vec
        self.euler_Psi, self.euler_Theta, self.euler_Phi = self.create_euler(args)
        self.vel_North, self.vel_East, self.vel_Down = self.create_vel(args)
        self.H_asl, self.pos_North, self.pos_East, self.Lat, self.Lon, self.pos_mpd_N, self.pos_mpd_E \
            = self.create_pos(args, map_data)
        self.H_map_grid, self.H_map = self.create_map(map_data)  # interpolated map according to position
        self.H_agl = self.H_asl - self.H_map
        self.pinpoint = PinPointCalc(self, map_data)  # calculate pinpoint coordinates

    @staticmethod
    def create_euler(args):
        """
        creating euler angels vectors
        assuming: straight flight
        """
        psi_vec = args.psi * np.ones(args.run_points)
        theta_vec = args.theta * np.ones(args.run_points)
        phi_vec = args.phi * np.ones(args.run_points)

        return psi_vec, theta_vec, phi_vec

    @staticmethod
    def create_vel(args):
        """
        creating velocity vectors
        assuming: straight flight, constant velocity
        """
        """
        # North axis
        ^
        |       /
        |  phi /
        |     /
        |    /
        |   /
        |  /
        +----------------> East axis
        """
        vel_north = args.avg_spd * cosd(args.psi) * np.ones(args.run_points)
        vel_east = args.avg_spd * sind(args.psi) * np.ones(args.run_points)
        vel_down = np.zeros(args.run_points)

        return vel_north, vel_east, vel_down

    def create_pos(self, args, map_data):
        """
        creating position vectors
        assuming: straight flight, constant velocity
        """
        h_asl = args.height * np.ones(args.run_points)

        # 501 arbitrary to avoid interpolation problems later
        x = 501
        pos_north = map_data.North[x] + self.vel_North * self.time_vec
        pos_east = map_data.East[x] + self.vel_East * self.time_vec

        pos_lat = np.linspace(args.lat, map_data.pos_final_lat, pos_north.size)
        pos_lon = np.linspace(args.lon, map_data.pos_final_lon, pos_east.size)

        pos_mpd_N = pos_north / pos_lat  # [m/deg]
        pos_mpd_E = pos_east / pos_lon  # [m/deg]

        assert pos_lat.min() > map_data.Lat.min() and pos_lat.max() < map_data.Lat.max(), "position latitude is out " \
                                                                                          "of bounds "
        assert pos_lon.min() > map_data.Lon.min() and pos_lon.max() < map_data.Lon.max(), "position longitude is out " \
                                                                                          "of bounds "

        return h_asl, pos_north, pos_east, pos_lat, pos_lon, pos_mpd_N, pos_mpd_E

    def create_map(self, map_data):
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
        interpolator = RegularGridInterpolator((map_data.Lat, map_data.Lon), map_data.map_grid)
        points = np.array(np.meshgrid(self.Lat, self.Lon)).T.reshape(-1, 2)
        h_map = interpolator(points).reshape(self.Lat.size, self.Lon.size)
        traj_heights = interpolator(np.vstack((self.Lat, self.Lon)).T)
        return h_map, traj_heights


class NoiseTraj(BaseTraj):
    """
        this class noising the trajectory
        the noise level declared in the input
    """
    def __init__(self, args, traj):
        super().__init__(args)
        self.run_length = args.run_points
        self.time = np.arange(self.run_length)
        self.noise_pos, self.noise_h_alt, self.noise_h_bar, self.noise_vel, self.noise_euler = self.get_noise(args)
        self.baro_bias = args.err_baro_bias
        self.pos_North, self.pos_East, self.Lat, self.Lon = self.generate_meas_pos(traj)
        self.pos_mpd_N, self.pos_mpd_E = self.create_pos_mpd()
        self.H_asl, self.H_agl = self.generate_meas_height(traj)
        self.vel_North, self.vel_East, self.vel_Down = self.generate_meas_vel(traj)
        self.euler_Psi, self.euler_Theta, self.euler_Phi = self.generate_meas_euler(traj)
        self.R_pinpoint = traj.pinpoint.R + rnd.normal(0, abs(self.noise_h_alt), traj.pinpoint.R.size)
        self.dN, self.dE = traj.pinpoint.dN, traj.pinpoint.dE
        self.H_map_pinpoint = traj.pinpoint.H_map + rnd.normal(0, abs(self.noise_h_bar), traj.pinpoint.H_map.size)

    @staticmethod
    def get_noise(args):
        return args.err_pos * rnd.randn(1), args.err_alt * rnd.randn(1), args.err_baro_noise * rnd.randn(1), \
               args.err_vel * rnd.randn(1), args.err_euler * rnd.rand(1)

    def generate_meas_pos_bu(self, traj):
        init_north = traj.pos_North[0] + self.noise_pos
        init_east = traj.pos_East[0] + self.noise_pos
        init_vel_east = traj.vel_East[0] + self.noise_vel
        init_vel_north = traj.vel_North[0] + self.noise_vel

        north = init_north + init_vel_north * np.arange(self.run_length)
        east = init_east + init_vel_east * np.arange(self.run_length)

        lat = north / traj.pos_mpd_N
        lon = east / traj.pos_mpd_E

        return north, east, lat, lon

    def generate_meas_pos(self, traj):
        north = traj.pos_North + rnd.normal(0, abs(self.noise_pos), traj.pos_North.size)
        east = traj.pos_East + rnd.normal(0, abs(self.noise_pos), traj.pos_East.size)
        east = traj.pos_East + rnd.normal(0, abs(self.noise_pos), traj.pos_East.size)
        lat = north / traj.pos_mpd_N
        lon = east / traj.pos_mpd_E

        return north, east, lat, lon

    def generate_meas_height(self, traj):
        # Above sea Level
        h_asl = traj.H_asl + rnd.normal(0, abs(self.noise_h_bar), self.run_length) + traj.vel_Down * self.time

        # Above ground level
        h_agl = traj.H_agl + rnd.normal(0, abs(self.noise_h_alt), self.run_length) + traj.vel_Down * self.time

        return h_asl, h_agl

    def generate_meas_vel(self, traj):
        north = traj.vel_North + rnd.normal(0, abs(self.noise_vel), traj.vel_North.size)
        east = traj.vel_East + rnd.normal(0, abs(self.noise_vel), traj.vel_East.size)
        down = traj.vel_Down + rnd.normal(0, abs(self.noise_vel), traj.vel_Down.size)

        return north, east, down

    def generate_meas_euler(self, traj):
        psi = traj.euler_Psi + rnd.normal(0, abs(self.noise_euler), traj.euler_Psi.size)
        theta = traj.euler_Theta + rnd.normal(0, abs(self.noise_euler), traj.euler_Theta.size)
        phi = traj.euler_Phi + rnd.normal(0, abs(self.noise_euler), traj.euler_Phi.size)

        return psi, theta, phi

    def create_pos_mpd(self):
        return self.pos_North / self.Lat, self.pos_East / self.Lon
