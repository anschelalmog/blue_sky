import pytest
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from unittest.mock import Mock
from src.noise_traj import NoiseTraj
from src.create_traj import CreateTraj


class TestNoiseTraj:
    @pytest.fixture
    def mock_true_traj(self):
        # Mock a CreateTraj instance with predefined attributes
        true_traj = Mock(spec=CreateTraj)
        true_traj.run_points = 100
        true_traj.time_vec = np.linspace(0, 99, 100)
        true_traj.mpd_north = 1  # Assuming 1 for simplicity
        true_traj.mpd_east = 1  # Assuming 1 for simplicity
        true_traj.pos = Mock()
        true_traj.vel = Mock()
        true_traj.euler = Mock()
        true_traj.pinpoint = Mock()

        # Assign mock arrays to pos, vel, euler attributes
        for attr in ['lat', 'lon', 'north', 'east', 'h_asl', 'h_agl', 'h_map']:
            setattr(true_traj.pos, attr, np.zeros(true_traj.run_points))

        for attr in ['north', 'east', 'down']:
            setattr(true_traj.vel, attr, np.zeros(true_traj.run_points))

        for attr in ['theta', 'psi', 'phi']:
            setattr(true_traj.euler, attr, np.zeros(true_traj.run_points))

        return true_traj

    @pytest.fixture
    def imu_errors(self):
        return {
            'euler_angles': 0.1,  # radians
            'velocity': 0.5,  # m/s
            'altimeter_noise': 1.0,  # meters
            'initial_position': 2.0,  # meters
            'barometer_noise': 0.5,  # meters
            'barometer_bias': 0.1,  # meters
        }

    def test_noise_euler_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_euler = np.copy(noise_traj.euler.theta)  # Assuming theta, psi, phi start as zeros

        noise_traj._noise_euler(imu_errors['euler_angles'], dist='normal')

        # Test that noise has been added
        assert not np.array_equal(noise_traj.euler.theta, original_euler)
        # Additional checks could include statistical properties of the noise

    def test_noise_velocity_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_vel = np.copy(noise_traj.vel.north)  # Assuming north, east, down start as zeros

        noise_traj._noise_velocity(imu_errors['velocity'], dist='normal')

        # Test that noise has been added
        assert not np.array_equal(noise_traj.vel.north, original_vel)
        # Additional checks could include statistical properties of the noise

    def test_noise_pinpoint_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_range = np.copy(noise_traj.pinpoint.range)  # Assuming range starts as zeros

        noise_traj._noise_pinpoint(imu_errors['altimeter_noise'], dist='normal')

        # Test that noise has been added
        assert not np.array_equal(noise_traj.pinpoint.range, original_range)
        # Additional checks could include statistical properties of the noise

    def test_noise_position_normal_distribution(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        original_pos_north = np.copy(noise_traj.pos.north)  # Assuming north, east start as zeros

        noise_traj._noise_position(imu_errors['initial_position'], imu_errors['barometer_noise'],
                                   imu_errors['barometer_bias'], imu_errors['altimeter_noise'], dist='normal')

        # Test that noise has been added
        assert not np.array_equal(noise_traj.pos.north, original_pos_north)

    def test_polynomial_fit_pos_with_noise(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.noise(imu_errors, dist='normal')

        # Test polynomial fit for the 'north' position component
        coeffs_north = np.polyfit(noise_traj.time_vec, noise_traj.pos.north, 1)
        poly_north = Polynomial(coeffs_north)

        # The test verifies that the first derivative (velocity) remains within expected bounds despite noise
        assert np.allclose(poly_north.deriv()(noise_traj.time_vec), noise_traj.vel.north, atol=5)

    def test_polynomial_fit_vel_with_noise(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.noise(imu_errors, dist='normal')

        # Test polynomial fit for the 'north' velocity component
        coeffs_vel_north = np.polyfit(noise_traj.time_vec, noise_traj.vel.north, 1)
        poly_vel_north = Polynomial(coeffs_vel_north)

        # The test verifies that the velocity trend remains linear despite noise
        assert np.allclose(poly_vel_north(noise_traj.time_vec), noise_traj.vel.north, atol=5)

    def test_polynomial_fit_euler_with_noise(self, mock_true_traj, imu_errors):
        np.random.seed(42)  # Ensuring consistent noise for the test
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.noise(imu_errors, dist='normal')

        coeffs_theta = np.polyfit(noise_traj.time_vec, noise_traj.euler.theta, 1)
        poly_theta = Polynomial(coeffs_theta)

        assert np.allclose(poly_theta(noise_traj.time_vec), noise_traj.euler.theta, atol=0.5)
