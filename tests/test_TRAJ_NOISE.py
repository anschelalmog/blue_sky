import pytest
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from unittest.mock import Mock
from src.noise_traj import NoiseTraj
from src.create_traj import CreateTraj


class TestNoiseTraj:
    """
    Test suite for the NoiseTraj class.

    This class contains a series of unit tests designed to verify the functionality and reliability of adding noise to a
    trajectory using the NoiseTraj class. It tests the addition of noise to various components of a trajectory,
    such as Euler angles, velocity, and position, and evaluates the ability of these noised components
    to be approximated by polynomial fits, ensuring the noise does not introduce unrealistic deviations.

    Fixtures:
    - mock_true_traj: Provides a mock instance of the CreateTraj class with predefined attributes, representing a "true"
                      trajectory without noise.
    - imu_errors: Supplies a dictionary of IMU error settings used to simulate realistic noise in the trajectory.

    Methods:
    - test_noise_euler_normal_distribution: Verifies that the addition of noise to Euler angles (theta, psi, phi)
                                            follows a normal distribution and modifies the original values as expected.
    - test_noise_velocity_normal_distribution: Ensures that the velocity components (north, east, down) are correctly
                                                noised with a normal distribution, affecting the original zero values.
    - test_noise_pinpoint_normal_distribution: Checks that noise added to pinpoint range measurements follows a normal
                                               distribution, altering the initial values.
    - test_noise_position_normal_distribution: Confirms that position components (north, east) are appropriately noised,
                                               deviating from their initial values according to the specified normal
                                               distribution noise parameters.
    - test_polynomial_fit_pos_with_noise: Assesses whether the noisy position data can still be approximated by a
                                          polynomial fit, indicating that the noise does not introduce unrealistic
                                          behavior in the trajectory's positional data.
    - test_polynomial_fit_vel_with_noise: Similar to the position test, this method verifies that the velocity
                                          components, even when noised, can be approximated by a polynomial,
                                          ensuring the noise maintains a realistic velocity profile.
    - test_polynomial_fit_euler_with_noise: Checks that Euler angles with added noise can be approximated by a
                                             polynomial fit, suggesting that the noisy angular data still exhibits
                                             a trend consistent with the original trajectory.
    """
    @pytest.fixture
    def mock_true_traj(self):
        true_traj = Mock(spec=CreateTraj)
        true_traj.run_points = 100
        true_traj.time_vec = np.linspace(0, 99, 100)
        true_traj.mpd_north = 1
        true_traj.mpd_east = 1

        # Mocking position, velocity, and euler as before
        true_traj.pos = Mock()
        true_traj.vel = Mock()
        true_traj.euler = Mock()

        # Additionally mocking the 'acc' attribute
        true_traj.acc = Mock()

        # Setting up attributes for pos, vel, euler, and now acc
        for attr in ['lat', 'lon', 'north', 'east', 'h_asl', 'h_agl', 'h_map']:
            setattr(true_traj.pos, attr, np.zeros(true_traj.run_points))

        for attr in ['north', 'east', 'down']:
            setattr(true_traj.vel, attr, np.zeros(true_traj.run_points))

        for attr in ['theta', 'psi', 'phi']:
            setattr(true_traj.euler, attr, np.zeros(true_traj.run_points))

        # Assuming 'acc' has similar attributes to 'vel' for this example
        for attr in ['north', 'east', 'down']:
            setattr(true_traj.acc, attr, np.zeros(true_traj.run_points))

        true_traj.pinpoint = Mock()

        return true_traj

    @pytest.fixture
    def imu_errors(self):
        return {
            'euler_angles': 0.1,  # radians
            'velocity': 20,  # m/s
            'altimeter_noise': 5,  # meters
            'initial_position': 20.0,  # meters
            'barometer_noise': 20,  # meters
            'barometer_bias': 10,  # meters
            'accelerometer': 0.01  # m/s^2
        }

    def test_noise_euler_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_euler = np.copy(noise_traj.euler.theta)

        noise_traj._noise_euler(imu_errors['euler_angles'], dist='normal')

        assert not np.array_equal(noise_traj.euler.theta, original_euler)

    def test_noise_velocity_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_vel = np.copy(noise_traj.vel.north)

        noise_traj._noise_velocity(imu_errors['velocity'], dist='normal')

        assert not np.array_equal(noise_traj.vel.north, original_vel)

    def test_noise_pinpoint_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_range = np.copy(noise_traj.pinpoint.range)

        noise_traj._noise_pinpoint(imu_errors['altimeter_noise'], dist='normal')

        assert not np.array_equal(noise_traj.pinpoint.range, original_range)

    def test_noise_position_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_pos_north = np.copy(noise_traj.pos.north)
        noise_traj._noise_position(imu_errors['initial_position'], imu_errors['barometer_noise'],
                                   imu_errors['barometer_bias'], imu_errors['altimeter_noise'], dist='normal')

        assert not np.array_equal(noise_traj.pos.north, original_pos_north)

    def test_polynomial_fit_pos_with_noise(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.noise(imu_errors, dist='normal')

        coeffs_north = np.polyfit(noise_traj.time_vec, noise_traj.pos.north, 1)
        poly_north = Polynomial(coeffs_north)

        assert np.allclose(poly_north.deriv()(noise_traj.time_vec), noise_traj.vel.north, atol=5)

    def test_polynomial_fit_vel_with_noise(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.noise(imu_errors, dist='normal')

        coeffs_vel_north = np.polyfit(noise_traj.time_vec, noise_traj.vel.north, 1)
        poly_vel_north = Polynomial(coeffs_vel_north)

        assert np.allclose(poly_vel_north(noise_traj.time_vec), noise_traj.vel.north, atol=5)

    def test_polynomial_fit_euler_with_noise(self, mock_true_traj, imu_errors):
        np.random.seed(42)
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.noise(imu_errors, dist='normal')

        coeffs_theta = np.polyfit(noise_traj.time_vec, noise_traj.euler.theta, 1)
        poly_theta = Polynomial(coeffs_theta)

        assert np.allclose(poly_theta(noise_traj.time_vec), noise_traj.euler.theta, atol=0.5)

    def test_simulate_drift(self, mock_true_traj):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.euler.bias = 0  # Assuming there's a bias attribute
        original_bias = noise_traj.euler.bias

        noise_traj._simulate_drift('euler.bias', drift_rate=0.01)

        assert noise_traj.euler.bias != original_bias