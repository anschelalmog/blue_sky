import numpy as np
from scipy.interpolate import interp1d, RegularGridInterpolator

from blue_sky.utils.math import sind, cosd, DCM
from blue_sky.utils.progress import progress_bar


def jac_north(psi, theta, phi):
    """
    Calculate the north component of the Jacobian for pinpoint calculations.

    Args:
        psi (float): Yaw angle in degrees
        theta (float): Pitch angle in degrees
        phi (float): Roll angle in degrees

    Returns:
        float: North Jacobian component
    """
    return cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi)


def jac_east(psi, theta, phi):
    """
    Calculate the east component of the Jacobian for pinpoint calculations.

    Args:
        psi (float): Yaw angle in degrees
        theta (float): Pitch angle in degrees
        phi (float): Roll angle in degrees

    Returns:
        float: East Jacobian component
    """
    return sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi)


class PinPoint:
    """
    Class for calculating pinpoint coordinates and terrain data.

    The pinpoint is the point where the vehicle's sensor beam intersects
    with the terrain, providing information about the ground.

    Attributes:
        range (numpy.ndarray): Range from vehicle to pinpoint
        delta_north (numpy.ndarray): North offset from vehicle to pinpoint
        delta_east (numpy.ndarray): East offset from vehicle to pinpoint
        lat (numpy.ndarray): Latitude of pinpoint
        lon (numpy.ndarray): Longitude of pinpoint
        h_map (numpy.ndarray): Height of terrain at pinpoint
    """
    def __init__(self, length):
        """
        Initialize pinpoint arrays.

        Args:
            length (int): Length of the trajectory
        """
        self.range = np.zeros(length)
        self.delta_north = np.zeros(length)
        self.delta_east = np.zeros(length)
        self.lat = np.zeros(length)
        self.lon = np.zeros(length)
        self.h_map = np.zeros(length)

    def calc(self, traj, map_data):
        """
        Calculate pinpoint data along the trajectory.

        This method determines where the vehicle sensor beam intersects with
        the terrain for each point in the trajectory.

        Args:
            traj: Trajectory data
            map_data: Map containing terrain information

        Returns:
            self: Updated PinPoint instance
        """
        # Initialize arrays
        self.range = np.zeros(traj.run_points)
        self.delta_north = np.zeros(traj.run_points)
        self.delta_east = np.zeros(traj.run_points)
        self.lat = np.zeros(traj.run_points)
        self.lon = np.zeros(traj.run_points)
        self.h_map = np.zeros(traj.run_points)

        latV, lonV = traj.pos.lat, traj.pos.lon
        psiV, thetaV, phiV = traj.euler.psi, traj.euler.theta, traj.euler.phi

        bar_desc = "Pinpoint calculation"
        total_iterations = traj.run_points

        for i in progress_bar(total_iterations, bar_desc):
            try:
                # Set up range vector
                dR = np.arange(0, traj.inits['height'])  # [m]
                lat, lon = latV[i], lonV[i]  # [deg]
                psi, theta, phi = psiV[i], thetaV[i], phiV[i]

                # Create rotation matrix and calculate offsets
                dcm = DCM(yaw=traj.euler.psi[i], pitch=traj.euler.theta[i], roll=traj.euler.phi[i])
                dNp, dEp = dR * dcm.rot_north(), dR * dcm.rot_east()  # [m]
                dLat, dLon = dNp / map_data.mpd['north'][i], dEp / map_data.mpd['east'][i]  # [deg]
                lat_tag, lon_tag = latV[i] + dLat, lonV[i] + dLon

                # Get terrain heights along beam path
                interpolator = RegularGridInterpolator(
                    (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
                traj_heights = interpolator(np.vstack((lat_tag, lon_tag)).T)

                # Calculate height differences
                dH_star = traj.pos.h_asl[i] - traj_heights  # above alleged ground
                dH_tag = dR * cosd(theta) * cosd(phi)

                # Find intersection point
                interpolated_height = interp1d(dH_star - dH_tag, dR)
                self.range[i] = interpolated_height(0).item()  # Range to pinpoint

                # Calculate offsets and coordinates
                self.delta_north[i] = self.range[i] * jac_north(psi, theta, phi) / traj.mpd_north[i]
                self.delta_east[i] = self.range[i] * jac_east(psi, theta, phi) / traj.mpd_east[i]

                # Pinpoint lat-lon coordinates
                self.lat[i] = lat + self.delta_north[i]
                self.lon[i] = lon + self.delta_east[i]

                # Get ground elevation at pinpoint
                self.h_map[i] = interpolator((self.lat[i], self.lon[i])).item()

            except ValueError as e:
                if "out of bounds" in str(e):
                    print(f"Interpolation error at index {i}: {e}")
                    # Continue with next iteration, leaving zeros in the arrays
                elif "A value in x_new is below the interpolation range" in str(e):
                    print(f"Interpolation range error at index {i}: {e}")
                    # Try to recover with a default value
                    self.range[i] = traj.inits['height'] / 2  # Approximate value
                    self.delta_north[i] = self.range[i] * jac_north(psi, theta, phi) / traj.mpd_north[i]
                    self.delta_east[i] = self.range[i] * jac_east(psi, theta, phi) / traj.mpd_east[i]
                    self.lat[i] = lat + self.delta_north[i]
                    self.lon[i] = lon + self.delta_east[i]
                else:
                    raise

        return self