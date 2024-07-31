import pytest
import numpy as np
from unittest.mock import MagicMock
from argparse import Namespace

from src.data_loaders import *
from src.create_traj import *
from src.noise_traj import *
from src.output_utils import *

from src.estimators import IEKF, IEKFParams


class TestIEKF:
    # .run(self.map_data, self.meas_traj)
    # self.run_errors, self.covariances = calc_errors_covariances(self.true_traj, self.estimation_results)

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

    """
        tests for flat surface, no error
    """

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

    def test_iekf_in_bounds_percentage_flat_surface_no_error(self, pos_min_error=80, vel_min_error=80, euler_min_error=80):
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

    """
        tests for flat surface, with error
    """

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
