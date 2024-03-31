import pytest
from unittest.mock import MagicMock
from src.estimators import IEKF, IEKFParams

# Mock classes for dependencies
class MockArgs:
    def __init__(self):
        self.run_points = 10
        self.time_vec = [i for i in range(10)]
        self.time_res = 1
        self.kf_state_size = 12

class MockMapData:
    pass

class MockMeas:
    pass

@pytest.fixture
def mock_args():
    return MockArgs()

@pytest.fixture
def mock_map_data():
    return MockMapData()

@pytest.fixture
def mock_meas():
    return MockMeas()

@pytest.fixture
def iekf_instance(mock_args):
    return IEKF(mock_args)

def test_initialization(iekf_instance):
    assert iekf_instance.run_points == 10
    assert iekf_instance.state_size == 12
    assert isinstance(iekf_instance.params, IEKFParams)

def test_run_method_updates_state_correctly(iekf_instance, mock_map_data, mock_meas):
    initial_state = iekf_instance.curr_state
    iekf_instance.run(mock_map_data, mock_meas)
    assert iekf_instance.curr_state == 9

def test_predict_state(iekf_instance, mock_meas):
    iekf_instance._predict_state = MagicMock()
    iekf_instance._predict_state(mock_meas)
    iekf_instance._predict_state.assert_called_with(mock_meas)


    def test_process_noise_covariance_Q(self, iekf_instance):
        # Test the value of the process noise covariance matrix Q
        iekf_instance._initialize_params()
        Q_expected = np.power(np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1e-7)
        assert np.allclose(iekf_instance.params.Q, Q_expected)

    def test_measurement_noise_covariance_R(self, iekf_instance):
        # Test the value of the measurement noise covariance matrix R
        iekf_instance._initialize_params()
        h_agl_meas = 500
        iekf_instance._calc_rc(h_agl_meas)
        Rc_expected = 100 + 125 + 175
        assert iekf_instance.params.Rc[iekf_instance.curr_state] == Rc_expected
        # Add assertions for Rfit and the total measurement noise covariance R

    def test_kalman_gain_K(self, iekf_instance):
        # Test the value of the Kalman gain K
        iekf_instance._initialize_params()
        P = np.eye(iekf_instance.state_size)
        iekf_instance._compute_gain(P)
        H = iekf_instance.params.H[:, iekf_instance.curr_state]
        R = iekf_instance.params.R[iekf_instance.curr_state]
        K_expected = P @ H.T / (H @ P @ H.T + R)
        assert np.allclose(iekf_instance.params.K[:, iekf_instance.curr_state], K_expected)

    def test_state_covariance_P(self, iekf_instance):
        # Test the value of the state covariance matrix P
        iekf_instance._initialize_params()
        P_init_expected = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)
        assert np.allclose(iekf_instance.params.P_est[:, :, 0], P_init_expected)
