import pytest
import numpy as np
from unittest.mock import Mock
from src.noise_traj import NoiseTraj
from src.create_traj import CreateTraj
import unittest

class TestNoiseTraj(unittest.TestCase):
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

    @pytest.fixture
    def noise_traj(self, mock_true_traj):
        return NoiseTraj(mock_true_traj)

    @pytest.fixture
    def bottom_up_noise_traj(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.add_noise(imu_errors, dist='normal', approach='bottom-up')
        return noise_traj

    @pytest.fixture
    def top_down_noise_traj(self, mock_true_traj, imu_errors):
        noise_traj = NoiseTraj(mock_true_traj)
        noise_traj.add_noise(imu_errors, dist='normal', approach='top-down')
        return noise_traj

    def test_add_noise_approach_distribution(self, mock_true_traj, imu_errors):
        """Tests if noise is added with correct approach and distribution settings."""
        noise_traj = NoiseTraj(mock_true_traj)

        noise_traj.add_noise(imu_errors, dist='normal', approach='bottom-up')
        assert noise_traj.approach == 'bottom-up'
        assert noise_traj.dist == 'normal'

        noise_traj.add_noise(imu_errors, dist='uniform', approach='top-down')
        assert noise_traj.approach == 'top-down'
        assert noise_traj.dist == 'uniform'

        with pytest.raises(AssertionError):
            noise_traj.add_noise(imu_errors, dist='invalid', approach='bottom-up')

        with pytest.raises(AssertionError):
            noise_traj.add_noise(imu_errors, dist='normal', approach='invalid')

    def test_add_noise_bottom_up_approach(self, bottom_up_noise_traj):
        """Tests if noise is added using bottom-up approach."""
        assert bottom_up_noise_traj.approach == 'bottom-up'
        assert bottom_up_noise_traj.dist == 'normal'

    def test_add_noise_top_down_approach(self, top_down_noise_traj):
        """Tests if noise is added using top-down approach."""
        assert top_down_noise_traj.approach == 'top-down'
        assert top_down_noise_traj.dist == 'normal'

    def test_noise_euler_normal_distribution(self, bottom_up_noise_traj, imu_errors):
        """Tests if noise added to Euler angles follows a normal distribution."""
        original_theta = np.copy(bottom_up_noise_traj.euler.theta)
        original_psi = np.copy(bottom_up_noise_traj.euler.psi)
        original_phi = np.copy(bottom_up_noise_traj.euler.phi)

        bottom_up_noise_traj._noise_euler(imu_errors['gyroscope'])

        assert not np.array_equal(bottom_up_noise_traj.euler.theta, original_theta)
        assert not np.array_equal(bottom_up_noise_traj.euler.psi, original_psi)
        assert not np.array_equal(bottom_up_noise_traj.euler.phi, original_phi)

    def test_noise_acc_normal_distribution(self, bottom_up_noise_traj, imu_errors):
        """Tests if noise added to acceleration follows a normal distribution."""
        original_north = np.copy(bottom_up_noise_traj.acc.north)
        original_east = np.copy(bottom_up_noise_traj.acc.east)
        original_down = np.copy(bottom_up_noise_traj.acc.down)

        bottom_up_noise_traj._noise_acc(imu_errors['accelerometer'])

        assert not np.array_equal(bottom_up_noise_traj.acc.north, original_north)
        assert not np.array_equal(bottom_up_noise_traj.acc.east, original_east)
        assert not np.array_equal(bottom_up_noise_traj.acc.down, original_down)

    def test_noise_velocity_normal_distribution(self, bottom_up_noise_traj, imu_errors):
        """Tests if noise added to velocity follows a normal distribution."""
        original_north = np.copy(bottom_up_noise_traj.vel.north)
        original_east = np.copy(bottom_up_noise_traj.vel.east)
        original_down = np.copy(bottom_up_noise_traj.vel.down)

        bottom_up_noise_traj._noise_velocity(imu_errors['velocity meter'])

        assert not np.array_equal(bottom_up_noise_traj.vel.north, original_north)
        assert not np.array_equal(bottom_up_noise_traj.vel.east, original_east)
        assert not np.array_equal(bottom_up_noise_traj.vel.down, original_down)

    def test_noise_position_normal_distribution(self, bottom_up_noise_traj, imu_errors):
        """Tests if noise added to position follows a normal distribution."""
        original_north = np.copy(bottom_up_noise_traj.pos.north)
        original_east = np.copy(bottom_up_noise_traj.pos.east)
        original_h_asl = np.copy(bottom_up_noise_traj.pos.h_asl)
        original_h_agl = np.copy(bottom_up_noise_traj.pos.h_agl)

        bottom_up_noise_traj._noise_position(imu_errors['position'], imu_errors['barometer'], imu_errors['altimeter'])

        assert not np.array_equal(bottom_up_noise_traj.pos.north, original_north)
        assert not np.array_equal(bottom_up_noise_traj.pos.east, original_east)
        assert not np.array_equal(bottom_up_noise_traj.pos.h_asl, original_h_asl)
        assert not np.array_equal(bottom_up_noise_traj.pos.h_agl, original_h_agl)

    def test_noise_pinpoint_normal_distribution(self, bottom_up_noise_traj, imu_errors):
        """Tests if noise added to pinpoint range follows a normal distribution."""
        original_range = np.copy(bottom_up_noise_traj.pinpoint.range)

        bottom_up_noise_traj._noise_pinpoint(imu_errors['altimeter'])

        assert not np.array_equal(bottom_up_noise_traj.pinpoint.range, original_range)

    def test_polynomial_fit_pos_with_noise(self, bottom_up_noise_traj):
        """Tests if the noisy position data can be approximated by a polynomial fit."""
        poly_fit_north = np.polyfit(bottom_up_noise_traj.time_vec, bottom_up_noise_traj.pos.north, 3)
        poly_fit_east = np.polyfit(bottom_up_noise_traj.time_vec, bottom_up_noise_traj.pos.east, 3)

        assert poly_fit_north is not None
        assert poly_fit_east is not None

    def test_polynomial_fit_vel_with_noise(self, bottom_up_noise_traj):
        """Tests if the noisy velocity data can be approximated by a polynomial fit."""
        poly_fit_north = np.polyfit(bottom_up_noise_traj.time_vec, bottom_up_noise_traj.vel.north, 3)
        poly_fit_east = np.polyfit(bottom_up_noise_traj.time_vec, bottom_up_noise_traj.vel.east, 3)

        assert poly_fit_north is not None
        assert poly_fit_east is not None

    def test_polynomial_fit_euler_with_noise(self, bottom_up_noise_traj):
        """Tests if the noisy Euler angles data can be approximated by a polynomial fit."""
        poly_fit_theta = np.polyfit(bottom_up_noise_traj.time_vec, bottom_up_noise_traj.euler.theta, 3)
        poly_fit_psi = np.polyfit(bottom_up_noise_traj.time_vec, bottom_up_noise_traj.euler.psi, 3)

        assert poly_fit_theta is not None
        assert poly_fit_psi is not None
