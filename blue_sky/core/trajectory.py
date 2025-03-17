import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

from blue_sky.core.base import BaseTraj
from blue_sky.core.pinpoint import PinPoint
from blue_sky.utils.math import cosd, sind, get_mpd, DCM


class CreateTraj(BaseTraj):
    """
    A class to create and handle trajectory data.

    This class generates a trajectory based on initial conditions and
    updates it with respect to terrain data.

    Attributes:
        run_points (int): Number of points in the trajectory
        time_vec (numpy.ndarray): Time values for each trajectory point
        inits (dict): Initial conditions and parameters
        pinpoint (PinPoint): Pinpoint calculation object
    """
    def __init__(self, args):
        """
        Initialize trajectory creator.

        Args:
            args: Configuration arguments
        """
        super().__init__(args.run_points)
        self.run_points = args.run_points
        self.time_vec = args.time_vec

        # Store initial conditions and parameters
        self.inits = {
            'lat': args.init_lat,
            'lon': args.init_lon,
            'height': args.init_height,
            'avg_spd': args.avg_spd,
            'acc_north': args.acc_north,
            'acc_east': args.acc_east,
            'acc_down': args.acc_down,
            'psi': args.psi,
            'theta': args.theta,
            'phi': args.phi,
            'psi_dot': args.psi_dot,
            'theta_dot': args.theta_dot,
            'phi_dot': args.phi_dot
        }

        self.pinpoint = None

    def create(self, map_data):
        """
        Create the full trajectory including position, velocity, and orientation.

        This method generates a complete trajectory based on the initial
        conditions and map data.

        Args:
            map_data: Map containing terrain information

        Returns:
            self: Updated trajectory
        """
        # Create basic motion components
        self._create_euler()
        self._create_acc()
        self._create_vel()
        self._create_pos()

        # Apply transformations and calculate terrain interactions
        self._apply_dcm(map_data)

        try:
            self._create_traj(map_data)
            # Calculate pinpoint data
            self.pinpoint = PinPoint(self.run_points).calc(self, map_data)
            # Calculate height above ground level
            self.pos.h_agl = self.pos.h_asl - self.pinpoint.h_map
        except ValueError as e:
            if "out of bounds" in str(e):
                print(f"Error in trajectory creation: {e}")
                # Try to recover with approximate values
                self.pos.h_map = np.ones(self.run_points) * 1000  # Default terrain height
                self.pos.h_agl = self.pos.h_asl - self.pos.h_map
            else:
                raise

        return self

    def _create_euler(self):
        """
        Create Euler angle vectors assuming linear change over time.
        """
        self.euler.psi = self.inits['psi'] + self.inits['psi_dot'] * self.time_vec
        self.euler.theta = self.inits['theta'] + self.inits['theta_dot'] * self.time_vec
        self.euler.phi = self.inits['phi'] + self.inits['phi_dot'] * self.time_vec

    def _create_acc(self):
        """
        Create acceleration vectors with constant values.
        """
        self.acc.north = self.inits['acc_north'] * np.ones(self.run_points)
        self.acc.east = self.inits['acc_east'] * np.ones(self.run_points)
        self.acc.down = self.inits['acc_down'] * np.ones(self.run_points)

    def _create_vel(self):
        """
        Create velocity vectors based on initial speed and acceleration.
        """
        # Initial velocity components
        self.vel.north = self.inits['avg_spd'] * cosd(self.inits['psi']) + self.acc.north * self.time_vec
        self.vel.east = self.inits['avg_spd'] * sind(self.inits['psi']) + self.acc.east * self.time_vec
        self.vel.down = self.inits['avg_spd'] * sind(self.inits['theta']) + self.acc.down * self.time_vec

    def _create_pos(self):
        """
        Create position vectors considering constant acceleration.
        """
        # Calculate meters per degree conversion factors
        self.mpd_north, self.mpd_east = get_mpd(self.inits['lat'])

        # Initial position in meters
        init_north = self.inits['lat'] * self.mpd_north
        init_east = self.inits['lon'] * self.mpd_east

        # Calculate positions with kinematic equations: X = X_0 + V_0 * t + 0.5 * a * t^2
        self.pos.north = (init_north + self.vel.north[0] * self.time_vec +
                          0.5 * self.inits['acc_north'] * self.time_vec ** 2)
        self.pos.east = (init_east + self.vel.east[0] * self.time_vec +
                         0.5 * self.inits['acc_east'] * self.time_vec ** 2)
        self.pos.h_asl = (self.inits['height'] + self.vel.down[0] -
                          0.5 * self.inits['acc_down'] * self.time_vec ** 2)

        # Convert back to lat/lon
        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east

        # Recalculate conversion factors at each position
        self.mpd_north, self.mpd_east = get_mpd(self.pos.lat)

        # Refine lat/lon calculations
        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east

    def _apply_dcm(self, map_data):
        """
        Apply the Direction Cosine Matrix to account for vehicle orientation.

        Args:
            map_data: Map data for terrain information
        """
        transformed_offsets = np.zeros((self.run_points, 3))

        for i in range(self.run_points):
            # Calculate IMU offset based on current height
            imu_offset = np.array([0, 0, -self.pos.h_asl[i]])

            # Create rotation matrix for current orientation
            dcm = DCM(yaw=self.euler.psi[i], pitch=self.euler.theta[i], roll=self.euler.phi[i])

            # Transform offset by rotation
            transformed_offset = dcm.matrix @ imu_offset
            transformed_offsets[i, :] = transformed_offset

            # Update position based on transformed offset
            self.pos.lat[i] += transformed_offset[0] / self.mpd_north[i]
            self.pos.lon[i] += transformed_offset[1] / self.mpd_east[i]

        # Update measured height map for new positions
        self._update_measured_height_map(map_data)

        return transformed_offsets

    def _update_measured_height_map(self, map_data):
        """
        Update height map measurements for the trajectory.

        Args:
            map_data: Map data containing terrain information
        """
        try:
            interpolator = RegularGridInterpolator(
                (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
            points = np.vstack((self.pos.lat, self.pos.lon)).T
            self.pos.h_map = interpolator(points)
        except ValueError as e:
            if "out of bounds" in str(e):
                print(f"Interpolation error in height map update: {e}")
                # Use default values
                self.pos.h_map = np.ones(self.run_points) * 1000
            else:
                raise

    def _create_traj(self, map_data):
        """
        Interpolate map data at trajectory points to calculate corresponding heights.

        Args:
            map_data: Map data containing terrain information
        """
        try:
            interpolator = RegularGridInterpolator(
                (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
            points = np.vstack((self.pos.lat, self.pos.lon)).T
            self.pos.h_map = interpolator(points)
        except ValueError as e:
            if "out of bounds" in str(e):
                print(f"Interpolation error in trajectory creation: {e}")
                # Try to recover
                min_lat = np.min(self.pos.lat)
                max_lat = np.max(self.pos.lat)
                min_lon = np.min(self.pos.lon)
                max_lon = np.max(self.pos.lon)

                # Check if map boundaries need to be extended
                map_data.update_map(min_lat - 0.1, min_lon - 0.1)
                map_data.update_map(max_lat + 0.1, max_lon + 0.1)

                # Try again with updated map
                interpolator = RegularGridInterpolator(
                    (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
                points = np.vstack((self.pos.lat, self.pos.lon)).T
                self.pos.h_map = interpolator(points)
            else:
                raise