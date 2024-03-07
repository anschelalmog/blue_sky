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

    def noise(self, imu_errors, dist='normal'):
        """
        Add noise to the trajectory based on specified distribution.

        :param imu_errors: Dictionary containing error settings.
        :param dist: String specifying the distribution type ('normal', 'uniform' or 'none').
        """
        if dist == 'none':
            print("----Applying no noise to the trajectory.----")
            return self

        if dist == 'normal':
            print("----Applying normal distribution noise to the trajectory.----")
        elif dist == 'uniform':
            print("----Applying uniform distribution noise to the trajectory.----")
        else:
            raise ValueError("dist must be either 'normal', 'uniform', or 'none'")

        self._noise_euler(imu_errors['euler_angles'], dist)
        self._noise_acc(imu_errors['accelerometer'], dist)
        self._noise_velocity(imu_errors['velocity'], dist)
        self._noise_pinpoint(imu_errors['altimeter_noise'], dist)
        self._noise_position(imu_errors['initial_position'], imu_errors['barometer_noise'],
                             imu_errors['barometer_bias'], imu_errors['altimeter_noise'], dist)

        return self

    def _noise_euler(self, error, dist):
        if dist == 'normal':
            noise = error * rnd.randn(self.run_points)
        else:  # uniform
            noise = error * rnd.uniform(-1, 1, size=self.run_points)

        self.euler.theta += noise
        self.euler.psi += noise
        self.euler.phi += noise

    def _noise_acc(self, acc_error, dist='normal'):
        """
        Apply noise to the acceleration measurements.

        :param acc_errors: Dictionary containing acceleration error settings for north, east, and down components.
        :param dist: String specifying the distribution type ('normal' or 'uniform').
        """
        if dist == 'normal':
            noise_north = acc_error * np.random.randn(self.run_points)
            noise_east = acc_error * np.random.randn(self.run_points)
            noise_down = acc_error * np.random.randn(self.run_points)
        elif dist == 'uniform':
            noise_north = acc_error * np.random.uniform(-1, 1, size=self.run_points)
            noise_east = acc_error * np.random.uniform(-1, 1, size=self.run_points)
            noise_down = acc_error * np.random.uniform(-1, 1, size=self.run_points)
        else:
            raise ValueError("Unsupported distribution type. Choose 'normal' or 'uniform'.")

        # Add noise to the acceleration components
        self.acc.north += noise_north
        self.acc.east += noise_east
        self.acc.down += noise_down
    def _noise_velocity(self, error, dist):
        if dist == 'normal':
            noise = error * rnd.randn(1)
        else:  # uniform
            noise = error * rnd.uniform(-1, 1)

        self.vel.north = self.vel.north[0] + noise * np.ones(self.run_points)
        self.vel.east = self.vel.east[0] + noise * np.ones(self.run_points)
        self.vel.down = self.vel.down[0] + noise * np.ones(self.run_points)

    def _noise_pinpoint(self, error, dist):
        if dist == 'normal':
            noise = error * rnd.randn(self.run_points)
        else:  # uniform
            noise = error * rnd.uniform(-1, 1, size=self.run_points)

        self.pinpoint.range += noise

    def _noise_position(self, pos_error, baro_noise, baro_bias, alt_error, dist):
        if dist == 'normal':
            pos_noise = pos_error * rnd.randn(1)
            h_asl_noise = baro_noise * rnd.randn(self.run_points)
            h_agl_noise = alt_error * rnd.randn(self.run_points)
        else:  # uniform
            pos_noise = pos_error * rnd.uniform(-1, 1)
            h_asl_noise = baro_noise * rnd.uniform(-1, 1, size=self.run_points)
            h_agl_noise = alt_error * rnd.uniform(-1, 1, size=self.run_points)

        self.pos.north = self.pos.north[0] + pos_noise + self.vel.north[0] * self.time_vec
        self.pos.east = self.pos.east[0] + pos_noise + self.vel.east[0] * self.time_vec
        self.pos.lat = self.pos.north / self.mpd_north
        self.pos.lon = self.pos.east / self.mpd_east

        self.pos.h_asl += baro_bias + h_asl_noise + self.vel.down[0] * self.time_vec
        self.pos.h_agl += h_agl_noise + self.vel.down[0] * self.time_vec
        self.h_map = self.pos.h_asl - self.pos.h_agl

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

        def _simulate_sensor_bias_drift(self, sensor_bias_attr, drift_rate, steps=None):
            """
            Simulate sensor bias drift over time using a random walk model.

            :param sensor_bias_attr: Attribute name (string) for the sensor bias to drift.
            :param drift_rate: The standard deviation of the drift step.
            :param steps: Number of steps for the drift simulation (optional, defaults to run_points).
            """
            if steps is None:
                steps = self.run_points

            # Generate random walk for bias drift
            drift = np.cumsum(drift_rate * np.random.randn(steps))

            # Apply the drift to the sensor bias
            bias_value = getattr(self, sensor_bias_attr)
            setattr(self, sensor_bias_attr, bias_value + drift)
