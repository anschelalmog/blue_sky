import numpy as np
import numpy.random as rnd
from src.base_traj import BaseTraj
from src.pinpoint_calc import PinPoint
from matplotlib import pyplot as plt


class NoiseTraj(BaseTraj):
    def __init__(self, true_traj):
        super().__init__(true_traj.run_points)
        self.run_points: int = true_traj.run_points
        self.time_vec = true_traj.time_vec
        self.mpd_north = true_traj.mpd_north
        self.mpd_east = true_traj.mpd_east
        self.approach: str = None
        self.dist: str = None
        self.true_traj = true_traj

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
        assert self.approach in ['bottom-up', 'top-down'], 'Invalid approach'

        self._noise_euler(imu_errors['gyroscope'])
        self._noise_acc(imu_errors['accelerometer'])
        self._noise_velocity(imu_errors['velocity meter'])
        self._noise_pinpoint(imu_errors['altimeter'])
        self._noise_position(imu_errors['position'], imu_errors['barometer'],
                             imu_errors['altimeter'])

        return self

    def _noise_euler(self, error):
        """
        Apply noise to the Euler angles

        Assuming that we have a gyroscope
        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
                      for bias rate and drift rate.
        """
        noise = self.__generate_noise(error['amplitude'], (3, self.run_points))
        if self.approach == 'bottom-up':
            noise = self.__apply_bias_and_drift(noise, error['bias'], error['drift'], (3, self.run_points))

        self.euler.theta += noise[0, :]
        self.euler.psi += noise[1, :]
        self.euler.phi += noise[2, :]

    def _noise_acc(self, error):
        """
        Apply noise to the acceleration measurements

        Assuming we have an accelerometer
        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
                      for bias rate and drift rate.
        """

        noise = self.__generate_noise(error['amplitude'], (3, self.run_points))
        if self.approach == 'bottom-up':
            noise = self.__apply_bias_and_drift(noise, error['bias'], error['drift'], (3, self.run_points))

        self.acc.north += noise[0, :]
        self.acc.east += noise[1, :]
        self.acc.down += noise[2, :]

    def _noise_velocity(self, error):
        """
        Apply noise to the velocity measurements

        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
         for bias rate and drift rate.

        """
        # Initialize noise based on distribution
        noise = self.__generate_noise(error['amplitude'], (3, self.run_points))
        if self.approach == 'bottom-up':
            noise = self.__apply_bias_and_drift(noise, error['bias'], error['drift'], (3, self.run_points))

        self.vel.north += noise[0] * np.ones(self.run_points)
        self.vel.east += noise[1] * np.ones(self.run_points)
        self.vel.down += noise[2] * np.ones(self.run_points)

    def _noise_pinpoint(self, error):
        """
           pinpoint range is based on the altimeter sensor
           this vector is the finalized measurements accumulating the noise
           in the bottom up version it is already noised
        """
        noise = self.__generate_noise(error['amplitude'], (self.run_points,))
        self.pinpoint.range += noise

    def _noise_position(self, pos_error, baro_error, alt_error):
        """
        Apply noise to the position measurements, supporting both bottom-up and top-down approaches.
        """

        # set noise
        init_pos_error_north = self.__generate_noise(pos_error['amplitude'], (1,))
        init_pos_error_east = self.__generate_noise(pos_error['amplitude'], (1,))
        north_noise = self.__generate_noise(pos_error['amplitude'], (self.run_points,))
        east_noise = self.__generate_noise(pos_error['amplitude'], (self.run_points,))
        h_asl_noise = self.__generate_noise(baro_error['amplitude'], (self.run_points,))
        h_agl_noise = self.__generate_noise(alt_error['amplitude'], (self.run_points,))

        methoed = self.approach
        self.approach = 'top-down'
        if self.approach == 'bottom-up':
            self.pos.north = self.pos.north[0] + init_pos_error_north + self.vel.north * self.time_vec
            self.pos.east = self.pos.east[0] + init_pos_error_east + self.vel.east * self.time_vec
            self.pos.h_asl = self.true_traj.pos.h_asl + h_asl_noise + self.vel.down * self.time_vec
            self.pos.h_agl = self.pos.h_agl[0] + h_agl_noise + self.vel.down * self.time_vec
        else:  # "top-down"
            self.pos.north += north_noise
            self.pos.north += east_noise
            self.pos.h_asl += h_asl_noise
            self.pos.h_map += h_agl_noise

        self.approach = methoed
        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east
        self.pos.h_map = self.pos.h_asl - self.pos.h_agl

    def __generate_noise(self, amplitude, shape):
        if self.dist == 'normal':
            return amplitude * rnd.randn(*shape)
        else:  # Uniform
            return amplitude * rnd.uniform(-1, 1, size=shape)

    @staticmethod
    def __apply_bias_and_drift(noise, bias, drift, shape):
        noise += bias * np.ones(shape)
        noise += np.cumsum(drift * rnd.randn(*shape), axis=1)
        return noise
