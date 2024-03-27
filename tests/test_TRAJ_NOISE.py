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

        # Mocking position, velocity, euler, and acc
        true_traj.pos = Mock()
        true_traj.vel = Mock()
        true_traj.euler = Mock()
        true_traj.acc = Mock()

        # Setting up attributes for pos, vel, euler, and acc
        for attr in ['lat', 'lon', 'north', 'east', 'h_asl', 'h_agl', 'h_map']:
            setattr(true_traj.pos, attr, np.zeros(true_traj.run_points))

        for attr in ['north', 'east', 'down']:
            setattr(true_traj.vel, attr, np.zeros(true_traj.run_points))
            setattr(true_traj.acc, attr, np.zeros(true_traj.run_points))

        for attr in ['theta', 'psi', 'phi']:
            setattr(true_traj.euler, attr, np.zeros(true_traj.run_points))

        true_traj.pinpoint = Mock()
        true_traj.pinpoint.range = np.zeros(true_traj.run_points)

        return true_traj

    @pytest.fixture
    def imu_errors(self):
        return {
            'gyroscope': {'amplitude': 0.1, 'bias': 0.01, 'drift': 0.001},
            'velocity meter': {'amplitude': 20, 'bias': 1, 'drift': 0.1},
            'altimeter': {'amplitude': 5},
            'position': {'amplitude': 20},
            'barometer': {'amplitude': 20, 'bias': 10, 'drift': 0.5},
            'accelerometer': {'amplitude': 0.01, 'bias': 0.005, 'drift': 0.0005}
        }

    def test_noise_euler_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_euler = np.copy(noise_traj.euler.theta)

        noise_traj._noise_euler(imu_errors['gyroscope'])

        assert not np.array_equal(noise_traj.euler.theta, original_euler)
        assert np.abs(np.mean(noise_traj.euler.theta - original_euler)) < 1
        assert np.abs(np.std(noise_traj.euler.theta - original_euler) - imu_errors['gyroscope']['amplitude']) < 1

    def test_noise_velocity_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_vel = np.copy(noise_traj.vel.north)

        noise_traj._noise_velocity(imu_errors['velocity meter'])

        assert not np.array_equal(noise_traj.vel.north, original_vel)
        assert np.abs(np.mean(noise_traj.vel.north - original_vel)) < 20
        assert np.abs(np.std(noise_traj.vel.north - original_vel) - imu_errors['velocity meter']['amplitude']) < 40

    def test_noise_pinpoint_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_range = np.copy(noise_traj.pinpoint.range)

        noise_traj._noise_pinpoint(imu_errors['altimeter'])

        assert not np.array_equal(noise_traj.pinpoint.range, original_range)
        assert np.abs(np.mean(noise_traj.pinpoint.range - original_range)) < 20
        assert np.abs(np.std(noise_traj.pinpoint.range - original_range) - imu_errors['altimeter']['amplitude']) < 40

    def test_noise_position_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_pos_north = np.copy(noise_traj.pos.north)

        noise_traj._noise_position(imu_errors['position'], imu_errors['barometer'], imu_errors['altimeter'])

        assert not np.array_equal(noise_traj.pos.north, original_pos_north)
        assert np.abs(np.mean(noise_traj.pos.north - original_pos_north)) < 20
        assert np.abs(np.std(noise_traj.pos.north - original_pos_north) - imu_errors['position']['amplitude']) < 40

    def test_polynomial_fit_pos_with_noise(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.add_noise(imu_errors, dist='normal')

        coeffs_north = np.polyfit(noise_traj.time_vec, noise_traj.pos.north, 1)
        poly_north = Polynomial(coeffs_north)

        assert np.allclose(poly_north.deriv()(noise_traj.time_vec), noise_traj.vel.north, atol=10e3)

    def test_polynomial_fit_vel_with_noise(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.add_noise(imu_errors, dist='normal')

        coeffs_vel_north = np.polyfit(noise_traj.time_vec, noise_traj.vel.north, 1)
        poly_vel_north = Polynomial(coeffs_vel_north)

        assert np.allclose(poly_vel_north(noise_traj.time_vec), noise_traj.vel.north, atol=10e3)

    def test_polynomial_fit_euler_with_noise(self, mock_true_traj, imu_errors):
        np.random.seed(42)
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.add_noise(imu_errors, dist='normal')

        coeffs_theta = np.polyfit(noise_traj.time_vec, noise_traj.euler.theta, 1)
        poly_theta = Polynomial(coeffs_theta)

        assert np.allclose(poly_theta(noise_traj.time_vec), noise_traj.euler.theta, atol=10e2)

    def test_add_noise_approach_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)

        noise_traj.add_noise(imu_errors, dist='normal', approach='bottom-up')
        assert noise_traj.approach == 'bottom-up'
        assert noise_traj.dist == 'normal'

        noise_traj.add_noise(imu_errors, dist='uniform', approach='top')
        assert noise_traj.approach == 'top'
        assert noise_traj.dist == 'uniform'

        with pytest.raises(AssertionError):
            noise_traj.add_noise(imu_errors, dist='invalid', approach='bottom-up')

        with pytest.raises(AssertionError):
            noise_traj.add_noise(imu_errors, dist='normal', approach='invalid')

    def test_add_noise_bottom_up_approach(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_euler = np.copy(noise_traj.euler.theta)
        original_acc = np.copy(noise_traj.acc.north)
        original_vel = np.copy(noise_traj.vel.north)
        original_range = np.copy(noise_traj.pinpoint.range)
        original_pos_north = np.copy(noise_traj.pos.north)

        noise_traj.add_noise(imu_errors, dist='normal', approach='bottom-up')

        assert not np.array_equal(noise_traj.euler.theta, original_euler)
        assert not np.array_equal(noise_traj.acc.north, original_acc)
        assert not np.array_equal(noise_traj.vel.north, original_vel)
        assert not np.array_equal(noise_traj.pinpoint.range, original_range)
        assert not np.array_equal(noise_traj.pos.north, original_pos_north)

    def test_add_noise_top_down_approach(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_euler = np.copy(noise_traj.euler.theta)
        original_acc = np.copy(noise_traj.acc.north)
        original_vel = np.copy(noise_traj.vel.north)
        original_range = np.copy(noise_traj.pinpoint.range)
        original_pos_north = np.copy(noise_traj.pos.north)

        noise_traj.add_noise(imu_errors, dist='normal', approach='top')

        assert not np.array_equal(noise_traj.euler.theta, original_euler)
        assert not np.array_equal(noise_traj.acc.north, original_acc)
        assert not np.array_equal(noise_traj.vel.north, original_vel)
        assert not np.array_equal(noise_traj.pinpoint.range, original_range)
        assert not np.array_equal(noise_traj.pos.north, original_pos_north)