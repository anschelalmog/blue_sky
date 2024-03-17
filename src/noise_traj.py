import numpy as np
import numpy.random as rnd
from src.base_traj import BaseTraj
from src.pinpoint_calc import PinPoint


class NoiseTraj(BaseTraj):
    def __init__(self, true_traj):
        super().__init__(true_traj.run_points)
        self.run_points = true_traj.run_points
        self.time_vec = true_traj.time_vec
        self.mpd_north = true_traj.mpd_north
        self.mpd_east = true_traj.mpd_east

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

    def noise(self, imu_errors, dist='normal', approach='bottom-up'):
        """
        Add noise to the trajectory based on specified distribution.

        :param imu_errors: Dictionary containing errors magnitudes
        :param dist: String specify kind 'normal', 'uniform' or 'none'.
        :param approach: String specifying the noise introduction approach ('bottom-up' or 'top-down').

        """
        assert dist in ['normal', 'uniform', 'none'], 'Invalid distribution'
        assert approach in ['bottom-up', 'top'], 'Invalid approach'

        if dist == 'none':
            print("----Applying no noise to the trajectory.----")
            return self
        elif dist == 'normal':
            print("----Applying normal distribution noise to the trajectory.----")
        else:  # 'uniform'
            print("----Applying uniform distribution noise to the trajectory.----")

        self._noise_euler(imu_errors['euler_angles'], dist, approach)
        self._noise_acc(imu_errors['accelerometer'], dist, approach)
        self._noise_velocity(imu_errors['velocity'], dist, approach)
        self._noise_pinpoint(imu_errors['altimeter_noise'], dist, approach)
        self._noise_position(imu_errors['initial_position'], imu_errors['barometer_noise'],
                             imu_errors['barometer_bias'], imu_errors['altimeter_noise'], dist, approach)

        return self

    def _noise_euler(self, error, dist, approach='bottom-up'):
        """
        Apply noise to the Euler angles, supporting both bottom-up and top-down approaches.

        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys
                      for bias rate and drift rate.
        :param dist: Distribution type ('normal' or 'uniform').
        :param approach: Approach type ('bottom-up' or 'top-down').
        """

        if dist == 'normal':
            noise = error['magnitude'] * rnd.randn(3, self.run_points)
        else:  # Uniform
            noise = error['magnitude'] * rnd.uniform(-1, 1, size=(3, self.run_points))

        if approach == 'bottom-up':
            # For bottom-up, include bias and drift simulation
            if 'bias_rate' in error:
                bias = error['bias_rate'] * np.ones((3, self.run_points))
                noise += bias

            if 'drift_rate' in error:
                drift = np.cumsum(error['drift_rate'] * rnd.randn(3, self.run_points), axis=1)
                noise += drift

        self.euler.theta += noise[0, :]
        self.euler.psi += noise[1, :]
        self.euler.phi += noise[2, :]

    def _noise_acc(self, error, dist, approach='bottom-up'):
        """
        Apply noise to the acceleration measurements, supporting both bottom-up and top-down approaches.

        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys for bias rate and drift rate.
        :param dist: Distribution type ('normal' or 'uniform').
        :param approach: Approach type ('bottom-up' or 'top-down').
        """

        if dist == 'normal':
            noise = error['magnitude'] * rnd.randn(3, self.run_points)
        else:  # Uniform
            noise = error['magnitude'] * rnd.uniform(-1, 1, size=(3, self.run_points))

        if approach == 'bottom-up':
            # For bottom-up, include bias and drift simulation
            if 'bias_rate' in error:
                bias = error['bias_rate'] * np.ones((3, self.run_points))
                noise += bias

            if 'drift_rate' in error:
                drift = np.cumsum(error['drift_rate'] * rnd.randn(3, self.run_points), axis=1)
                noise += drift

        self.acc.north += noise[0, :]
        self.acc.east += noise[1, :]
        self.acc.down += noise[2, :]

    def _noise_velocity(self, error, dist, approach):
        """
        Apply noise to the velocity measurements, supporting both bottom-up and top-down approaches.

        :param error: Dictionary containing the magnitude of the noise, and for the bottom-up approach, additional keys for bias rate and drift rate.
        :param dist: Distribution type ('normal' or 'uniform').
        :param approach: Approach type ('bottom-up' or 'top-down').
        """
        # Initialize noise based on distribution
        if dist == 'normal':
            noise = error['magnitude'] * rnd.randn(3, 1)
        else:  # Uniform
            noise = error['magnitude'] * rnd.uniform(-1, 1, size=(3, 1))

        if approach == 'bottom-up':
            # For bottom-up, include bias and drift simulation
            if 'bias_rate' in error:
                # Simulate constant bias over time
                bias = error['bias_rate'] * np.ones((3, 1))
                noise += bias

            if 'drift_rate' in error:
                # Simulate bias drift using a random walk model
                drift = np.cumsum(error['drift_rate'] * rnd.randn(3, 1), axis=1)
                noise += drift

        # Apply the final noise to the velocity components
        # Ensure noise is broadcasted correctly over all points
        self.vel.north += noise[0] * np.ones(self.run_points)
        self.vel.east += noise[1] * np.ones(self.run_points)
        self.vel.down += noise[2] * np.ones(self.run_points)

    def _noise_pinpoint(self, error, dist, approach):
        if approach == 'top-down':
            if dist == 'normal':
                noise = error * rnd.randn(self.run_points)
            else:  # uniform
                noise = error * rnd.uniform(-1, 1, size=self.run_points)
        # pinpoint range is based on the altimeter sensor
        # this vector is the finalized measurements accumulating the noise
        # in the bottom up version it is already noised
        else:  # 'bottom - up'
            noise = 0

        self.pinpoint.range += noise

    def _noise_position(self, pos_error, baro_noise, baro_bias, alt_error, dist, approach='bottom-up'):
        """
        Apply noise to the position measurements, supporting both bottom-up and top-down approaches.

        :param pos_error: Error magnitude for position noise.
        :param baro_noise: Noise magnitude for barometric altitude measurements.
        :param baro_bias: Constant bias for barometric altitude.
        :param alt_error: Error magnitude for altimeter noise.
        :param dist: Distribution type ('normal' or 'uniform').
        :param approach: Approach type ('bottom-up' or 'top-down').
        """
        # Generate base noise based on distribution
        if dist == 'normal':
            pos_noise = pos_error * rnd.randn(1)
            h_asl_noise = baro_noise * rnd.randn(self.run_points)
            h_agl_noise = alt_error * rnd.randn(self.run_points)
        else:  # Uniform
            pos_noise = pos_error * rnd.uniform(-1, 1)
            h_asl_noise = baro_noise * rnd.uniform(-1, 1, size=self.run_points)
            h_agl_noise = alt_error * rnd.uniform(-1, 1, size=self.run_points)

        if approach == 'bottom-up':
            # For bottom-up, include bias and drift for barometric altitude
            # Assume bias and drift for position are included in pos_noise, h_asl_noise, and h_agl_noise calculation
            h_asl_noise += baro_bias  # Apply constant bias to barometric altitude noise

        # Apply the final noise to position and altitude components
        self.pos.north += pos_noise + self.vel.north[0] * self.time_vec
        self.pos.east += pos_noise + self.vel.east[0] * self.time_vec
        self.pos.lat += self.pos.north / self.mpd_north  # Assuming lat/lon calculations are required after noise application
        self.pos.lon += self.pos.east / self.mpd_east

        self.pos.h_asl += h_asl_noise + self.vel.down[0] * self.time_vec
        self.pos.h_agl += h_agl_noise + self.vel.down[0] * self.time_vec
        self.h_map = self.pos.h_asl - self.pos.h_agl  # Update map height based on new altitudes

    def _noise_random_walk(self, attribute, step_error, steps=None):
        """
        Simulate random walk noise for a specified attribute.

        :param attribute: Attribute name (string) to apply the noise to.
        :param step_error: Standard deviation of the step error.
        :param steps: Number of steps for the random walk (optional, defaults to run_points).
        """
        if steps is None:
            steps = self.run_points

        # Generate random walk noise
        noise = np.cumsum(step_error * np.random.randn(steps))

        # Apply the noise to the specified attribute
        attr_value = getattr(self, attribute)
        setattr(self, attribute, attr_value + noise)

    def _noise_gauss_markov(self, attribute, error, correlation_time, time_step):
        """
        Simulate errors using a first-order Gauss-Markov process.

        :param attribute: Attribute name (string) to apply the noise to.
        :param error: Standard deviation of the process noise.
        :param correlation_time: Correlation time of the process.
        :param time_step: Time step between consecutive points.
        """
        # Initialize variables
        beta = 1 / correlation_time
        steps = self.run_points
        noise = np.zeros(steps)

        # Generate Gauss-Markov noise
        for i in range(1, steps):
            noise[i] = np.exp(-beta * time_step) * noise[i - 1] + \
                       np.sqrt((1 - np.exp(-2 * beta * time_step)) * error ** 2) * np.random.randn()

        # Apply the noise to the specified attribute
        attr_value = getattr(self, attribute)
        setattr(self, attribute, attr_value + noise)

        # Example usage:
        # noise_traj = NoiseTraj(true_traj)
        # noise_traj._noise_random_walk('pos.north', step_error=0.1)
        # noise_traj._noise_gauss_markov('vel.north', error=0.1, correlation_time=50, time_step=1)

    def _simulate_drift(self, sensor_bias_attr, drift_rate, steps=None):
        """
        Simulate sensor bias drift over time using a random walk model.

        :param sensor_bias_attr: Attribute name (string) for the sensor bias to drift.
        :param drift_rate: The standard deviation of the drift step.
        :param steps: Number of steps for the drift simulation (optional, defaults to run_points).
        """
        if steps is None:
            steps = self.run_points

        drift = np.cumsum(drift_rate * np.random.randn(steps))

        bias_value = getattr(self, sensor_bias_attr)
        setattr(self, sensor_bias_attr, bias_value + drift)
