import pytest
from src.data_loaders import set_settings
from src.estimators import IEKF  # replace 'yourmodule' with the actual module where IEKF is defined

# Fixture for IEKF instance setup
@pytest.fixture
def iekf_instance():
    args = set_settings()
    iekf = IEKF(args)
    return iekf

def test_run_method(mocker, iekf_instance):
    # Mock methods
    mock_initialize_traj = mocker.patch.object(IEKF, '_initialize_traj')
    mock_initialize_params = mocker.patch.object(IEKF, '_initialize_params')
    mock_predict_state = mocker.patch.object(IEKF, '_predict_state')
    mock_predict_covariance = mocker.patch.object(IEKF, '_predict_covariance')
    mock_pinpoint_coordinates = mocker.patch.object(IEKF, '_pinpoint_coordinates')
    mock_find_slopes = mocker.patch.object(IEKF, '_find_slopes')
    mock_calc_rc = mocker.patch.object(IEKF, '_calc_rc')
    mock_compute_gain = mocker.patch.object(IEKF, '_compute_gain')
    mock_estimate_covariance = mocker.patch.object(IEKF, '_estimate_covariance')
    mock_update_estimate_state = mocker.patch.object(IEKF, '_update_estimate_state')

    # Test data
    map_data = MagicMock()

    meas = MagicMock()

    # Run the method under test
    estimation_results = iekf_instance.run(map_data, meas)

    # Assertions
    mock_initialize_traj.assert_called_once_with(meas)
    mock_initialize_params.assert_called_once()
    mock_predict_state.assert_called()
    mock_predict_covariance.assert_called()
    mock_pinpoint_coordinates.assert_called()
    mock_find_slopes.assert_called()
    mock_calc_rc.assert_called()
    mock_compute_gain.assert_called()
    mock_estimate_covariance.assert_called()
    mock_update_estimate_state.assert_called()

    # Add more detailed checks as needed
