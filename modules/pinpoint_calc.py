import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d, RegularGridInterpolator
from modules.utils import sind, cosd


def jac_north(psi, theta, phi):
    return cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi)


def jac_east(psi, theta, phi):
    return sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi)


class PinPoint:
    def __init__(self, length):
        self.range = np.zeros(length)
        self.delta_north = np.zeros(length)
        self.delta_east = np.zeros(length)
        self.lat = np.zeros(length)
        self.lon = np.zeros(length)
        self.h_map = np.zeros(length)

    def calc(self, traj, map_data):
        """
        Calculating the true range from the vehicle to the pinpoint, and true
        pinpoint coordinates, for each point on the trajectory repeat required iterations
        :param map_data: map class instance, containing the map grid
        :param traj: true_traj class instance, containing the trajectory
        """
        self.range = np.zeros(traj.run_points)
        self.delta_north = np.zeros(traj.run_points)
        self.delta_east = np.zeros(traj.run_points)
        self.lat = np.zeros(traj.run_points)
        self.lon = np.zeros(traj.run_points)
        self.h_map = np.zeros(traj.run_points)
        latV, lonV = traj.pos.lat, traj.pos.lon
        psiV, thetaV, phiV = traj.euler.psi, traj.euler.theta, traj.euler.phi

        bar_desc = "pinpoint calculation"
        for i, lat in enumerate(tqdm(traj.pos.lat, desc=bar_desc)):
            dR = np.arange(0, traj.init_height + 1000)  # [m]
            lat, lon = latV[i], lonV[i]  # [deg]
            psi, theta, phi = psiV[i], thetaV[i], phiV[i]

            dNp, dEp = dR * jac_north(psi, theta, phi), dR * jac_east(psi, theta, phi)  # [m]
            dLat, dLon = dNp / map_data.mpd_north[i], dEp / map_data.mpd_east[i]  # [deg]
            lat_tag, lon_tag = lat + dLat, lon + dLon

            # height map data for the alleged pinpoint ( for each lat, lon along dR)
            interpolator = RegularGridInterpolator((map_data.ax_lat, map_data.ax_lon), map_data.grid)
            traj_heights = interpolator(np.vstack((lat_tag, lon_tag)).T)
            dH_star = traj.pos.h_asl[i] - traj_heights  # above alleged ground

            dH_tag = dR * cosd(theta) * cosd(phi)
            interpolated_height = interp1d(dH_star - dH_tag, dR)

            self.range[i] = interpolated_height(0).item()  # Range to pinpoint

            self.delta_north[i] = interpolated_height(0) * jac_north(psi, theta, phi) / traj.mpd_north[i]
            self.delta_east[i] = interpolated_height(0) * jac_east(psi, theta, phi) / traj.mpd_east[i]

            # pinpoint lat - lon coordinates
            self.lat[i] = lat + self.delta_north[i]
            self.lon[i] = lon + self.delta_east[i]

            # Ground Elevation from Map at PinPoint % height map data for the pinpoint
            self.h_map[i] = interpolator((self.lat[i], self.lon[i])).item()

        return self
