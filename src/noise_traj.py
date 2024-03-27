import numpy as np
import numpy.random as rnd
from src.base_traj import BaseTraj
from src.pinpoint_calc import PinPoint


class NoiseTraj(BaseTraj):
    def __init__(self, true_traj):
        super().__init__(true_traj.run_points)
        self.run_points: int = true_traj.run_points
        self.time_vec = true_traj.time_vec
        self.mpd_north = true_traj.mpd_north
        self.mpd_east = true_traj.mpd_east
        self.approach: str = None
        self.dist: str = None

        # Copy attributes of pos, vel, and euler
        for attr in ['pos', 'vel', 'euler', 'acc']:
            self_obj = getattr(self, attr)
            true_traj_obj = getattr(true_traj, attr)
            for sub_attr in vars(true_traj_obj):
                setattr(self_obj, sub_attr, getattr(true_traj_obj, sub_attr))

        self.pinpoint = PinPoint(self.run_points)
        for pin_attr in vars(true_traj.pinpoint):
            setattr(self.pinpoint, pin_attr, getattr(true_traj.pinpoint, pin_attr))

        self.pos.h_agl = true_traj.pos.h_asl - true_traj.pos.h_map

    def add_noise(self, imu_errors, dist='normal', approach='bottom-up'):
        """
        Add noise to the trajectory based on specified distribution.

        :param imu_errors: Dictionary containing errors magnitudes
        :param dist: String specify kind 'normal', 'uniform' or 'none'.
        :param approach: String specifying the noise introduction approach ('bottom-up' or 'top-down').

        """
        self.approach = approach
        self.dist = dist
        assert self.dist in ['normal', 'uniform', 'none'], 'Invalid distribution'
        assert self.approach in ['bottom-up', 'top'], 'Invalid approach'

        if self.dist == 'none':
            print("----Applying no noise to the trajectory.----")
            return self
        elif self.dist == 'normal':
            print("----Applying normal distribution noise to the trajectory.----")
        else:  # 'uniform'
            print("----Applying uniform distribution noise to the trajectory.----")

        self._noise_euler(imu_errors['gyroscope'])
        self._noise_acc(imu_errors['accelerometer'])
        self._noise_velocity(imu_errors['velocity meter'])
        self._noise_pinpoint(imu_errors['altimeter'])
        self._noise_position(imu_errors['position'], imu_errors['barometer'], imu_errors['altimeter'])

        return self

    def _noise_euler(self, error):
        """
        Apply noise to the Euler angles, supporting both bottom-up and top-down approaches.

        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
                      for bias rate and drift rate.
        """

        if self.dist == 'normal':
            noise = error['amplitude'] * rnd.randn(3, self.run_points)
        else:  # Uniform
            noise = error['amplitude'] * rnd.uniform(-1, 1, size=(3, self.run_points))

        if self.approach == 'bottom-up':
            # add bias and
            noise += error['bias'] * np.ones((3, self.run_points))
            noise += np.cumsum(error['drift'] * rnd.randn(3, self.run_points), axis=1)

        self.euler.theta += noise[0, :]
        self.euler.psi += noise[1, :]
        self.euler.phi += noise[2, :]

    def _noise_acc(self, error):
        """
        Apply noise to the acceleration measurements, supporting both bottom-up and top-down approaches.
        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
                      for bias rate and drift rate.
        """

        if self.dist == 'normal':
            noise = error['amplitude'] * rnd.randn(3, self.run_points)
        else:  # Uniform
            noise = error['amplitude'] * rnd.uniform(-1, 1, size=(3, self.run_points))

        if self.approach == 'bottom-up':
            noise += error['bias'] * np.ones((3, self.run_points))
            noise += np.cumsum(error['drift'] * rnd.randn(3, self.run_points), axis=1)

        self.acc.north += noise[0, :]
        self.acc.east += noise[1, :]
        self.acc.down += noise[2, :]

    def _noise_velocity(self, error):
        """
        Apply noise to the velocity measurements, supporting both bottom-up and top-down approaches.

        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
         for bias rate and drift rate.

        """
        # Initialize noise based on distribution
        if self.dist == 'normal':
            noise = error['amplitude'] * rnd.randn(3, 1)
        else:  # Uniform
            noise = error['amplitude'] * rnd.uniform(-1, 1, size=(3, 1))

        if self.approach == 'bottom-up':
            noise += error['bias'] * np.ones((3, 1))
            noise += np.cumsum(error['drift'] * rnd.randn(3, 1), axis=1)

        self.vel.north += noise[0] * np.ones(self.run_points)
        self.vel.east += noise[1] * np.ones(self.run_points)
        self.vel.down += noise[2] * np.ones(self.run_points)

        """
        # build the velocities from the accelerometer and gyroscope
        self.vel.north = np.cumsum(self.acc.north * np.cos(self.euler.theta) * np.cos(self.euler.psi) -
                                   self.acc.east * np.sin(self.euler.psi) +
                                   self.acc.down * np.sin(self.euler.theta) * np.cos(self.euler.psi)) + vel_noise[0, :]

        self.vel.east = np.cumsum(self.acc.north * np.cos(self.euler.theta) * np.sin(self.euler.psi) +
                                  self.acc.east * np.cos(self.euler.psi) +
                                  self.acc.down * np.sin(self.euler.theta) * np.sin(self.euler.psi)) + vel_noise[1, :]

        self.vel.down = np.cumsum(-self.acc.north * np.sin(self.euler.theta) +
                                  self.acc.down * np.cos(self.euler.theta)) + vel_noise[2, :]
        """

    def _noise_pinpoint(self, error):
        if self.approach == 'top-down':
            if self.dist == 'normal':
                noise = error * rnd.randn(self.run_points)
            else:  # uniform
                noise = error * rnd.uniform(-1, 1, size=self.run_points)
        else:  # 'bottom-up'
            """ 
                pinpoint range is based on the altimeter sensor
                this vector is the finalized measurements accumulating the noise
                in the bottom up version it is already noised
            """
            noise = 0

        self.pinpoint.range += noise

    def _noise_position(self, position_error, barometer_error, altimeter_error):
        """
        Apply noise to the position measurements, supporting both bottom-up and top-down approaches.

        :param position_error: Dictionary containing the position error parameters.
        :param barometer_error: Dictionary containing the barometer error parameters.
        :param altimeter_error: Dictionary containing the altimeter error parameters.
        """
        pass
        # Generate base noise based on distribution
        if self.dist == 'normal':
            pos_noise = position_error['amplitude'] * rnd.randn(2, self.run_points)
            h_asl_noise = barometer_error['amplitude'] * rnd.randn(self.run_points)
            h_agl_noise = altimeter_error['amplitude'] * rnd.randn(self.run_points)
        else:  # Uniform
            pos_noise = position_error['amplitude'] * rnd.uniform(-1, 1, size=(2, self.run_points))
            h_asl_noise = barometer_error['amplitude'] * rnd.uniform(-1, 1, size=self.run_points)
            h_agl_noise = altimeter_error['amplitude'] * rnd.uniform(-1, 1, size=self.run_points)

        if self.approach == 'bottom-up':
            # For bottom-up, include bias and drift for position, barometer, and altimeter
            pos_noise[0, :] += position_error['bias'][0] + np.cumsum(
                position_error['drift'][0] * rnd.randn(self.run_points))
            pos_noise[1, :] += position_error['bias'][1] + np.cumsum(
                position_error['drift'][1] * rnd.randn(self.run_points))
            h_asl_noise += barometer_error['bias'] + np.cumsum(barometer_error['drift'] * rnd.randn(self.run_points))
            h_agl_noise += altimeter_error['bias'] + np.cumsum(altimeter_error['drift'] * rnd.randn(self.run_points))

        # Apply the final noise to position and altitude components
        self.pos.north += pos_noise[0, :] + self.vel.north * self.time_vec
        self.pos.east += pos_noise[1, :] + self.vel.east * self.time_vec
        self.pos.lat += self.pos.north / self.mpd_north
        self.pos.lon += self.pos.east / self.mpd_east

        self.pos.h_asl += h_asl_noise + self.vel.down * self.time_vec
        self.pos.h_agl += h_agl_noise + self.vel.down * self.time_vec
        self.pos.h_map = self.pos.h_asl - self.pos.h_agl  # Update map height based on new altitudes
