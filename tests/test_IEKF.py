import pytest
import numpy as np
import unittest
from unittest.mock import MagicMock, patch
from argparse import Namespace

from BLUE_SKY.data_loaders import *
from BLUE_SKY.create_traj import *
from BLUE_SKY.noise_traj import *
from BLUE_SKY.output_utils import *
from BLUE_SKY.estimators import IEKF, IEKFParams


class TestIEKF:
    @pytest.fixture
    def default_args(self):
        return Namespace(
            traj_from_file=False, errors_from_table=True, traj_path='', plot_results=True, noise_type='normal',
            maps_dir='Map',
            map_level=1, time_init=0, time_end=100, time_res=0.1, kf_type='IEKF', kf_state_size=12, imu_errors=None,
            init_lat=37.5, init_lon=21.5, init_height=5000, avg_spd=250, psi=45, theta=0, phi=0, acc_north=0,
            acc_east=0,
            acc_down=0, psi_dot=0, theta_dot=0, phi_dot=0, run_points=1000, time_vec=np.arange(0, 100, 0.1), map_res=3,
            results_folder='D:\\python_project\\blue_sky\\out'
        )

    @pytest.fixture(autouse=True)
    def setup(self, default_args):
        self.args = default_args

    @pytest.fixture
    def mock_iekf(self):
        """Create a mock IEKF instance for testing."""
        args = MagicMock()
        args.run_points = 100
        args.time_vec = np.arange(0, 10, 0.1)
        args.kf_state_size = 12
        iekf = IEKF(args)
        iekf.params = MagicMock()
        iekf.params.SN = np.zeros(args.run_points)
        iekf.params.SE = np.zeros(args.run_points)
        iekf.params.Rfit = np.zeros(args.run_points)
        return iekf

    def run_iekf_setup_no_error(self, mock=True):
        """Helper method to run the setup steps for IEKF."""
        self.errors = IMUErrors(self.args.imu_errors)
        self.map_data = Map().load(self.args, mock=mock)
        self.true_traj = CreateTraj(self.args).create(self.map_data)
        self.meas_traj = NoiseTraj(self.true_traj).add_noise(self.errors.imu_errors)
        self.estimation_results = IEKF(self.args)

    def run_iekf_setup_with_error(self, mock=True):
        self.errors = IMUErrors(self.args.imu_errors)

        self.errors.set_imu_error('altimeter', amplitude=20)

        self.map_data = Map().load(self.args, mock=mock)
        self.true_traj = CreateTraj(self.args).create(self.map_data)
        self.meas_traj = NoiseTraj(self.true_traj).add_noise(self.errors.imu_errors)
        self.estimation_results = IEKF(self.args)

    @pytest.mark.parametrize("mock", [False, True])
    def test_setup(self, mock):
        self.run_iekf_setup_no_error(mock=mock)
        assert self.map_data is not None
        assert self.true_traj is not None
        assert self.meas_traj is not None
        assert self.estimation_results is not None

    def test_setup_errors(self):
        self.run_iekf_setup_no_error(mock=True)
        self.errors.set_imu_error('altimeter', amplitude=10, drift=10, bias=10)
        assert self.errors.imu_errors['altimeter']['amplitude'] == 10
        assert self.errors.imu_errors['altimeter']['drift'] == 10
        assert self.errors.imu_errors['altimeter']['bias'] == 10

    ########################################
    "Tests for flat surface, no error"     #
    ########################################

    def test_iekf_errors_close_to_zero_flat_surface_no_error(self, pos_thr=1, vel_thr=1, euler_thr=1):
        """
        Verify that errors after running the IEKF are close to zero.

        Explanation:
        If the surface is flat and there are no IMU errors, Kalman gains should be negligible,
        resulting in minimal updates to the state estimates. According to Kalman filter theory,
        when predictions perfectly match measurements, Kalman gains are minimal.

        This test:
        1. Sets up and runs the IEKF.
        2. Computes errors between true and estimated trajectories.
        3. Asserts that position, velocity, and Euler angles errors are within acceptable thresholds.
        """
        self.run_iekf_setup_no_error(mock=True)
        self.estimation_results.run(self.map_data, self.meas_traj)

        run_errors, covariances = calc_errors_covariances(self.meas_traj, self.estimation_results)
        length = self.args.run_points

        # Assert position errors are close to zero
        assert np.all(np.abs(run_errors.pos.lat[10:length - 10]) < pos_thr)
        assert np.all(np.abs(run_errors.pos.lon[10:length - 10]) < pos_thr)
        assert np.all(np.abs(run_errors.pos.h_asl[10:length - 10]) < pos_thr)

        # Assert velocity errors are close to zero
        assert np.all(np.abs(run_errors.vel.north[10:length - 10]) < vel_thr)
        assert np.all(np.abs(run_errors.vel.east[10:length - 10]) < vel_thr)
        assert np.all(np.abs(run_errors.vel.down[10:length - 10]) < vel_thr)

        # Assert Euler angles errors are close to zero
        assert np.all(np.abs(run_errors.euler.psi[10:length - 10]) < euler_thr)
        assert np.all(np.abs(run_errors.euler.theta[10:length - 10]) < euler_thr)
        assert np.all(np.abs(run_errors.euler.phi[10:length - 10]) < euler_thr)

    def test_iekf_in_bounds_percentage_flat_surface_no_error(self, pos_min_error=80, vel_min_error=80,
                                                             euler_min_error=80):
        """
        Verify that the errors are in 3sigma at least above the given percentage.
        """
        self.run_iekf_setup_no_error(mock=True)
        self.estimation_results.run(self.map_data, self.meas_traj)

        run_errors, covariances = calc_errors_covariances(self.meas_traj, self.estimation_results)
        length = self.args.run_points

        # Assert position errors are close to zero
        assert run_errors.metrics['pos']['north']['error_bound_percentage'] > pos_min_error
        assert run_errors.metrics['pos']['east']['error_bound_percentage'] > pos_min_error
        assert run_errors.metrics['pos']['north']['error_bound_percentage'] > pos_min_error

        # Assert velocity errors are close to zero
        assert run_errors.metrics['vel']['north']['error_bound_percentage'] > vel_min_error
        assert run_errors.metrics['vel']['east']['error_bound_percentage'] > vel_min_error
        assert run_errors.metrics['vel']['down']['error_bound_percentage'] > vel_min_error

        # Assert Euler angles errors are close to zero
        assert run_errors.metrics['euler']['psi']['error_bound_percentage'] > euler_min_error
        assert run_errors.metrics['euler']['theta']['error_bound_percentage'] > euler_min_error
        assert run_errors.metrics['euler']['phi']['error_bound_percentage'] > euler_min_error

    #######################################
    "Tests for flat surface, with error"  #
    #######################################
    def test_iekf_in_bounds_percentage_flat_surface_with_error(self):
        """
        Verify that the errors are within 3-sigma bounds for at least the specified percentage.
        Report all metrics that are outside the boundary.
        """
        MIN_ERROR_PERCENTAGES = {
            'pos': 80,
            'vel': 80,
            'euler': 80
        }

        self.run_iekf_setup_with_error(mock=True)
        self.estimation_results.run(self.map_data, self.meas_traj)

        run_errors, _ = calc_errors_covariances(self.meas_traj, self.estimation_results)

        errors = []
        out_of_bounds_metrics = []

        for metric, min_error in MIN_ERROR_PERCENTAGES.items():
            components = ['north', 'east', 'down'] if metric != 'pos' else ['north', 'east', 'h_asl']
            for component in components:
                if metric == 'euler':
                    component = {'north': 'psi', 'east': 'theta', 'down': 'phi'}[component]

                error_percentage = run_errors.metrics[metric][component]['error_bound_percentage']

                if error_percentage <= min_error:
                    errors.append(
                        f"{metric.capitalize()} {component} error ({error_percentage:.2f}%) is below minimum ({min_error}%)")
                    out_of_bounds_metrics.append((metric, component, error_percentage))

        # Report all out-of-bounds metrics
        if out_of_bounds_metrics:
            print("\nMetrics outside the boundary:")
            for metric, component, percentage in out_of_bounds_metrics:
                print(f"  {metric.capitalize()} {component}: {percentage:.2f}%")

        # Assert all errors at once using pytest
        if errors:
            pytest.fail("\n".join(errors))

    ######################################
    "Testing _find_slopes() method"      #
    ######################################

    def test_find_slopes(self):
        """
        Test _find_slopes method with different map types.
        First test with no error and expect exact values,
        then with error and expect minimal values.
        """
        mock_maps = Mock_Maps()

        # Test parameters
        lat, lon = 37.5, 21.5
        p_pre = np.eye(12) * 100
        i = 50  # Assuming this is the current state

        expected_values = {
            'flat_map': {'SN': 0, 'SE': 0, 'Rfit': 0},
            'linear_map': {'SN': -0.0027, 'SE': 0.0059, 'Rfit': 0},
            'wavy_map': {'SN': -0.87, 'SE': 0.94, 'Rfit': 100},
            'complex_map': {'SN': -1.74, 'SE': 1.88, 'Rfit': 1000}
        }

        for map_name, map_data in mock_maps.__dict__.items():
            if not map_name.endswith('_map'):
                continue

            with patch('scipy.interpolate.interp2d') as mock_interp2d:
                def mock_interpolate(x, y):
                    return np.array([map_data.grid[int((y - 37) * 100), int((x - 21) * 100)]])

                mock_interp2d.return_value = mock_interpolate

                # Calculate slopes
                SN, SE, Rfit = self._calculate_slopes(lat, lon, p_pre, map_data, i)

                # Assert values
                np.testing.assert_allclose(SN, expected_values[map_name]['SN'], rtol=1e-2,
                                           err_msg=f"SN mismatch for {map_name}")
                np.testing.assert_allclose(SE, expected_values[map_name]['SE'], rtol=1e-2,
                                           err_msg=f"SE mismatch for {map_name}")
                assert Rfit >= expected_values[map_name]['Rfit'], f"Rfit too small for {map_name}"

    def _calculate_slopes(self, lat, lon, p_pre, map_data, i):
        dP = 100  # distance increments in [m]
        delPmap = np.array([dP / map_data.mpd['north'][i], dP / map_data.mpd['east'][i]])  # [deg]

        # max number of points in each direction
        maxP = np.sqrt(max(p_pre[0][0], p_pre[1][1]))
        KP = 3
        NC = np.ceil(max(KP, 2 * np.ceil(KP * maxP / dP) + 1) / 2)
        idx = int((NC - 1) / 2)  # indices

        # create lat lon vectors according to grid indices
        pos_offset = np.arange(-idx, idx + 1)
        lat_vec, lon_vec = delPmap[0] * pos_offset + lat, delPmap[1] * pos_offset + lon
        xp, yp = np.meshgrid(lon_vec, lat_vec)
        xp, yp = xp[0, :], yp[:, 0]

        # scaling factors for slope calc
        sx2 = sy2 = (dP ** 2) * 2 * NC * np.sum(np.power(np.arange(1, idx + 1), 2))

        # interpolate elevation data
        interpolator = interp2d(map_data.axis['lon'], map_data.axis['lat'], map_data.grid)
        ref_elevation = float(interpolator(lon, lat))
        grid_elevations = interpolator(xp, yp)

        # calculate slopes in x and y directions
        syh = dP * np.dot(pos_offset, grid_elevations - ref_elevation).sum()
        sxh = dP * np.dot(grid_elevations - ref_elevation, pos_offset).sum()

        SN, SE = sxh / sx2, syh / sy2

        # calculate the Error over the grid
        MP = (2 * idx + 1) ** 2  # number of points in the mesh grid
        In = np.sum((dP * (SN * pos_offset[:, np.newaxis] + SE * pos_offset) -
                     grid_elevations + ref_elevation) ** 2)
        Rfit = In / (MP - 1)

        return SN, SE, Rfit


class Mock_Maps:
    def __init__(self):
        self.flat_map = self._create_map(self._flat_grid)
        self.linear_map = self._create_map(self._linear_grid)
        self.wavy_map = self._create_map(self._wavy_grid)
        self.complex_map = self._create_map(self._complex_grid)

    def _create_map(self, grid_func):
        map_data = MagicMock()
        map_data.axis = {
            'lat': np.linspace(37, 38, 101),
            'lon': np.linspace(21, 22, 101)
        }
        map_data.mpd = {
            'north': np.linspace(110000, 112000, 100),
            'east': np.linspace(84000, 86000, 100)
        }
        x, y = np.meshgrid(map_data.axis['lon'], map_data.axis['lat'])
        map_data.grid = grid_func(x, y)
        return map_data

    @staticmethod
    def _flat_grid(x, y):
        return np.full_like(x, 1000)  # Constant elevation of 1000m

    @staticmethod
    def _linear_grid(x, y):
        slope_lon = 500  # 500 meters per degree longitude
        slope_lat = -300  # -300 meters per degree latitude
        base_elevation = 1000  # meters
        return base_elevation + slope_lon * (x - x.min()) + slope_lat * (y - y.min())

    @staticmethod
    def _wavy_grid(x, y):
        return 1000 + 100 * np.sin(5*x) + 100 * np.cos(5*y)

    @staticmethod
    def _complex_grid(x, y):
        return 1000 + 500 * np.sin(2*x) * np.cos(2*y) + 300 * np.abs(np.sin(5*x) * np.cos(5*y))
