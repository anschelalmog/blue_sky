import numpy as np
import numpy.random as rnd

from ..core.base import BaseTraj
from ..core.pinpoint import PinPoint



class NoiseTraj(BaseTraj):
    """
    A class to add simulated noise to a trajectory.

    This class models realistic sensor imperfections by adding different
    types of noise to a true trajectory, simulating real-world measurements.

    Attributes:
        run_points (int): Number of points in the trajectory
        time_vec (numpy.ndarray): Time vector for the trajectory
        mpd_north (numpy.ndarray): Meters per degree in north direction
        mpd_east (numpy.ndarray): Meters per degree in east direction
        approach (str): Approach for noise introduction ('bottom-up' or 'top-down')
        dist (str): Noise distribution type ('normal', 'uniform', or 'none')
        true_traj: The original trajectory without noise
    """
    def __init__(self, true_traj):
        """
        Initialize a noisy trajectory based on a true trajectory.

        Args:
            true_traj: The original true trajectory to add noise to
        """
        super().__init__(true_traj.run_points)
        self.run_points = true_traj.run_points
        self.time_vec = true_traj.time_vec
        self.mpd_north = true_traj.mpd_north
        self.mpd_east = true_traj.mpd_east
        self.approach = None
        self.dist = None
        self.true_traj = true_traj

        # Copy attributes of pos, vel, and euler
        for attr in ['pos', 'vel', 'euler', 'acc']:
            self_obj = getattr(self, attr)
            true_traj_obj = getattr(true_traj, attr)
            for sub_attr in vars(true_traj_obj):
                setattr(self_obj, sub_attr, getattr(true_traj_obj, sub_attr))

        # Initialize pinpoint
        self.pinpoint = PinPoint(self.run_points)
        for pin_attr in vars(true_traj.pinpoint):
            setattr(self.pinpoint, pin_attr, getattr(true_traj.pinpoint, pin_attr))

        # Calculate height above ground level
        self.pos.h_agl = true_traj.pos.h_asl - true_traj.pos.h_map

    def add_noise(self, imu_errors, dist='normal', approach='bottom-up'):
        """
        Add noise to the trajectory based on specified distribution and approach.

        This method applies noise to various components of the trajectory including:
        - Euler angles (orientation)
        - Accelerations
        - Velocities
        - Pinpoint measurements
        - Position coordinates

        Args:
            imu_errors (dict): Dictionary containing error magnitudes for each sensor
            dist (str): Noise distribution type - 'normal', 'uniform', or 'none'
            approach (str): Noise introduction approach - 'bottom-up' or 'top-down'

        Returns:
            self: The noisy trajectory instance

        Raises:
            AssertionError: If the distribution or approach parameters are invalid
        """
        self.approach = approach
        self.dist = dist

        # Validate parameters
        assert self.dist in ['normal', 'uniform', 'none'], 'Invalid distribution'
        assert self.approach in ['bottom-up', 'top-down'], 'Invalid approach'

        # Apply noise to different components
        self._noise_euler(imu_errors['gyroscope'])
        self._noise_acc(imu_errors['accelerometer'])
        self._noise_velocity(imu_errors['velocity meter'])
        self._noise_pinpoint(imu_errors['altimeter'])
        self._noise_position(imu_errors['position'], imu_errors['barometer'],
                             imu_errors['altimeter'])

        return self

    def _noise_euler(self, error):
        """
        Apply noise to the Euler angles (orientation).

        Args:
            error (dict): Dictionary containing the magnitude of the noise, bias, and drift
        """
        noise = self.__generate_noise(error['amplitude'], (3, self.run_points))
        if self.approach == 'bottom-up':
            noise = self.__apply_bias_and_drift(noise, error['bias'], error['drift'], (3, self.run_points))

        self.euler.theta += noise[0, :]
        self.euler.psi += noise[1, :]
        self.euler.phi += noise[2, :]

    def _noise_acc(self, error):
        """
        Apply noise to the acceleration measurements.

        Args:
            error (dict): Dictionary containing the magnitude of the noise, bias, and drift
        """
        noise = self.__generate_noise(error['amplitude'], (3, self.run_points))
        if self.approach == 'bottom-up':
            noise = self.__apply_bias_and_drift(noise, error['bias'], error['drift'], (3, self.run_points))

        self.acc.north += noise[0, :]
        self.acc.east += noise[1, :]
        self.acc.down += noise[2, :]

    def _noise_velocity(self, error):
        """
        Apply noise to the velocity measurements.

        Args:
            error (dict): Dictionary containing the magnitude of the noise, bias, and drift
        """
        noise = self.__generate_noise(error['amplitude'], (3, self.run_points))
        if self.approach == 'bottom-up':
            noise = self.__apply_bias_and_drift(noise, error['bias'], error['drift'], (3, self.run_points))

        self.vel.north += noise[0] * np.ones(self.run_points)
        self.vel.east += noise[1] * np.ones(self.run_points)
        self.vel.down += noise[2] * np.ones(self.run_points)

    def _noise_pinpoint(self, error):
        """
        Apply noise to the pinpoint range measurements.

        This simulates altimeter sensor errors in measuring distance to the ground.

        Args:
            error (dict): Dictionary containing the amplitude of the noise
        """
        noise = self.__generate_noise(error['amplitude'], (self.run_points,))
        self.pinpoint.range += noise

    def _noise_position(self, pos_error, baro_error, alt_error):
        """
        Apply noise to the position measurements.

        This method supports both bottom-up and top-down approaches.
        - Bottom-up: Builds position from velocity with initial error
        - Top-down: Adds direct noise to position components

        Args:
            pos_error (dict): Position error parameters
            baro_error (dict): Barometer error parameters
            alt_error (dict): Altimeter error parameters
        """
        # Generate noise components
        init_pos_error_north = self.__generate_noise(pos_error['amplitude'], (1,))
        init_pos_error_east = self.__generate_noise(pos_error['amplitude'], (1,))
        north_noise = self.__generate_noise(pos_error['amplitude'], (self.run_points,))
        east_noise = self.__generate_noise(pos_error['amplitude'], (self.run_points,))
        h_asl_noise = self.__generate_noise(baro_error['amplitude'], (self.run_points,))
        h_agl_noise = self.__generate_noise(alt_error['amplitude'], (self.run_points,))

        # Store original approach and temporarily set to top-down for application
        method = self.approach
        self.approach = 'top-down'

        if method == 'bottom-up':
            # Bottom-up: Build position from initial error + velocity integration
            self.pos.north = self.pos.north[0] + init_pos_error_north + self.vel.north * self.time_vec
            self.pos.east = self.pos.east[0] + init_pos_error_east + self.vel.east * self.time_vec
            self.pos.h_asl = self.true_traj.pos.h_asl + h_asl_noise + self.vel.down * self.time_vec
            self.pos.h_agl = self.pos.h_agl[0] + h_agl_noise + self.vel.down * self.time_vec
        else:  # "top-down"
            # Top-down: Add noise directly to each position component
            self.pos.north += north_noise
            self.pos.east += east_noise
            self.pos.h_asl += h_asl_noise
            self.pos.h_map += h_agl_noise

        # Restore original approach
        self.approach = method

        # Update derived position components
        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east
        self.pos.h_map = self.pos.h_asl - self.pos.h_agl

    def __generate_noise(self, amplitude, shape):
        """
        Generate noise according to specified distribution.

        Args:
            amplitude (float): Amplitude of the noise
            shape (tuple): Shape of the noise array

        Returns:
            numpy.ndarray: Generated noise array
        """
        if self.dist == 'normal':
            return amplitude * rnd.randn(*shape)
        elif self.dist == 'uniform':
            return amplitude * rnd.uniform(-1, 1, size=shape)
        else:  # 'none'
            return np.zeros(shape)

    @staticmethod
    def __apply_bias_and_drift(noise, bias, drift, shape):
        """
        Apply bias and drift to noise.

        Args:
            noise (numpy.ndarray): Base noise array
            bias (float): Constant bias value
            drift (float): Drift rate for random walk
            shape (tuple): Shape of the noise array

        Returns:
            numpy.ndarray: Noise with bias and drift applied
        """
        # Add constant bias
        noise += bias * np.ones(shape)

        # Add random walk drift (accumulating error)
        noise += np.cumsum(drift * rnd.randn(*shape), axis=1)

        return noise