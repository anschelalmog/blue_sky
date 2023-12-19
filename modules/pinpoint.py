import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator
from tqdm import tqdm
from .utils import cosd, sind


def north_jac(psi, theta, phi):
    part1 = cosd(psi) * sind(theta) * cosd(phi)
    part2 = sind(psi) * sind(phi)
    return part1 + part2


def east_jac(psi, theta, phi):
    part1 = sind(psi) * sind(theta) * cosd(phi)
    part2 = cosd(psi) * sind(phi)
    return part1 - part2


class PinPointCalc:
    def __init__(self, traj, map_data):
        self.dR_range = 6000
        self.dN, self.dE, self.R, self.Lat, self.Lon, self.H_map = self.calc_pinpoint(map_data, traj)

    def calc_pinpoint(self, map_data, traj):
        """
        Calculated the true range from the vehicle to the pinpoint, and true
        pinpoint coordinates, for each point on the trajectory repeat required iterations
        :param map_data: map class instance, containing the map grid
        :param traj: true_traj class instance, containing the trajectory
        """
        dN_p = np.zeros(traj.Lon.size)
        dE_p = np.zeros(traj.Lon.size)
        R_pinpoint = np.zeros(traj.Lon.size)
        Lat_pp = np.zeros(traj.Lon.size)
        Lon_pp = np.zeros(traj.Lon.size)
        H_map_pp = np.zeros(traj.Lon.size)
        # for all ranges, calculate the latitude's and longitude's delta (from position to assumed pinpoint)
        bar_description = "PinPoint interpolating iterations"
        for i, lat in enumerate(tqdm(traj.Lat, desc=bar_description)):
            dR = np.arange(self.dR_range)
            lat, lon = traj.Lat[i], traj.Lon[i]  # range vector in small intervals [m]
            psi, theta, phi = traj.euler_Psi[i], traj.euler_Theta[i], traj.euler_Phi[i]  # % actual position [deg]

            dnp = dR * north_jac(psi, theta, phi)  # [m]
            dep = dR * east_jac(psi, theta, phi)  # [m]

            d_lat = dnp / traj.pos_North[i]  # [deg]
            d_lon = dep / traj.pos_East[i]  # [deg]

            lat_tag, lon_tag = lat + d_lat, lon + d_lon

            # height map data for the alleged pinpoint ( for each lat, lon along dR)
            interpolator = RegularGridInterpolator((map_data.Lat, map_data.Lon), map_data.map_grid)
            traj_heights = interpolator(np.vstack((lat_tag, lon_tag)).T)
            dH_star = traj.H_asl[i] - traj_heights  # above alleged ground

            dH_tag = dR * cosd(theta) * cosd(phi)
            interpolated_height = interp1d(dH_star - dH_tag, dR)
            R_pinpoint[i] = interpolated_height(0).item()  # Range to pinpoint

            dN_p[i] = interpolated_height(0) * north_jac(psi, theta, phi) / traj.pos_North[i]
            dE_p[i] = interpolated_height(0) * east_jac(psi, theta, phi) / traj.pos_East[i]

            # pinpoint lat - lon coordinates
            Lat_pp[i] = lat + dN_p[i]
            Lon_pp[i] = lon + dE_p[i]

            # Ground Elevation from Map at PinPoint % height map data for the pinpoint
            H_map_pp[i] = interpolator((Lat_pp[i], Lon_pp[i])).item()

        return dN_p, dE_p, R_pinpoint, Lat_pp, Lon_pp, H_map_pp
