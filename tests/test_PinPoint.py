import pytest
import numpy as np
from src.pinpoint_calc import PinPoint
from src.create_traj import CreateTraj
from src.data_loaders import Map


class TestPinPoint:
    @pytest.fixture
    def mock_args(self):
        # Create sample trajectory data for testing
        class Args:
            def __init__(self):
                self.maps_dir = 'Map'
                self.map_res = 3
                self.map_level = 1
                self.run_points = 100
                self.time_vec = np.linspace(0, 10, 100)
                self.time_end = 100
                self.init_lat = 37.5
                self.init_lon = 21.5
                self.init_height = 500
                self.avg_spd = 250
                self.acc_north = 0
                self.acc_east = 0
                self.acc_down = 0
                self.psi = 45
                self.theta = 0
                self.phi = 0
                self.psi_dot = 0
                self.theta_dot = 0
                self.phi_dot = 0
                self.results_folder = 'out'  # Add the missing attribute

        args = Args()
        return args

    @pytest.fixture
    def mock_traj(self, map_data, mock_args):
        traj = CreateTraj(mock_args)
        traj._create_euler()
        traj._create_acc()
        traj._create_vel()
        traj._create_pos()
        traj._create_traj(map_data[0])
        return traj

    @pytest.fixture
    def loaded_map(self, mock_args, mock=False):
        # Load map data from a file or database
        map_data = Map()
        map_data.load(mock_args, mock=mock)
        return map_data

    @pytest.mark.parametrize("mock", [True, False])
    def test_load_map(self, mock_args, mock):
        map_data = Map().load(mock_args, mock=mock)

        assert map_data.meta is not None, "Map metadata not initialized"
        assert map_data.axis is not None, "Map axis not initialized"
        assert map_data.bounds is not None, "Map bounds not initialized"
        assert map_data.mpd is not None, "Map mpd not initialized"
        assert map_data.grid is not None, "Map grid not initialized"

    @pytest.mark.parametrize("mock", [True, False])
    def test_pinpoint_calc_output_shape(self, mock_args, mock):
        """
        This test verifies that the output arrays of PinPoint.calc() have the expected
        shapes based on the number of run points in the trajectory.
        It ensures that the method returns the correct number of values for each output variable.
        """
        map_data = Map().load(mock_args, mock=mock)
        trajectory = CreateTraj(mock_args)
        trajectory._create_euler()
        trajectory._create_acc()
        trajectory._create_vel()
        trajectory._create_pos()
        trajectory._create_traj(map_data)
        trajectory.pinpoint = PinPoint(trajectory.run_points)
        trajectory.pinpoint.calc(trajectory, map_data)

        assert mock_traj.pinpoint.range.shape == (mock_traj.run_points,), \
            f"Expected range shape {(mock_traj.run_points,)}, but got {mock_traj.pinpoint.range.shape}"
        assert mock_traj.pinpoint.delta_north.shape == (mock_traj.run_points,), \
            f"Expected delta_north shape {(mock_traj.run_points,)}, but got {mock_traj.pinpoint.delta_north.shape}"
        assert mock_traj.pinpoint.delta_east.shape == (mock_traj.run_points,), \
            f"Expected delta_east shape {(mock_traj.run_points,)}, but got {mock_traj.pinpoint.delta_east.shape}"
        assert mock_traj.pinpoint.lat.shape == (mock_traj.run_points,), \
            f"Expected lat shape {(mock_traj.run_points,)}, but got {mock_traj.pinpoint.lat.shape}"
        assert mock_traj.pinpoint.lon.shape == (mock_traj.run_points,), \
            f"Expected lon shape {(mock_traj.run_points,)}, but got {mock_traj.pinpoint.lon.shape}"
        assert mock_traj.pinpoint.h_map.shape == (mock_traj.run_points,), \
            f"Expected h_map shape {(mock_traj.run_points,)}, but got {mock_traj.pinpoint.h_map.shape}"

