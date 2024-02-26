import pytest
from unittest.mock import MagicMock
from src.iekf import IEKF, IEKFParams

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
