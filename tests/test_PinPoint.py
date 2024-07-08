import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.pinpoint_calc import PinPoint
from src.create_traj import CreateTraj
from src.data_loaders import Map


class TestPinPoint:
    @pytest.fixture
    def mock_args(self):
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
                self.init_height = 5000
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
                self.results_folder = 'out'

        args = Args()
        return args

    @pytest.fixture
    def mock_traj(self, mock_args, loaded_map):
        traj = CreateTraj(mock_args)
        traj.create(loaded_map)
        return traj

    @pytest.fixture
    def loaded_map(self, mock_args):
        map_data = Map()
        map_data.load(mock_args)
        return map_data

    def test_load_map(self, mock_args):
        map_data = Map().load(mock_args)

        assert map_data.meta is not None, "Map metadata not initialized"
        assert map_data.axis is not None, "Map axis not initialized"
        assert map_data.bounds is not None, "Map bounds not initialized"
        assert map_data.mpd is not None, "Map mpd not initialized"
        assert map_data.grid is not None, "Map grid not initialized"

    def test_pinpoint_calc_output_shape(self, mock_args, loaded_map, mock_traj):
        trajectory = mock_traj

        assert trajectory.pinpoint.range.shape == (trajectory.run_points,), \
            f"Expected range shape {(trajectory.run_points,)}, but got {trajectory.pinpoint.range.shape}"
        assert trajectory.pinpoint.delta_north.shape == (trajectory.run_points,), \
            f"Expected delta_north shape {(trajectory.run_points,)}, but got {trajectory.pinpoint.delta_north.shape}"
        assert trajectory.pinpoint.delta_east.shape == (trajectory.run_points,), \
            f"Expected delta_east shape {(trajectory.run_points,)}, but got {trajectory.pinpoint.delta_east.shape}"
        assert trajectory.pinpoint.lat.shape == (trajectory.run_points,), \
            f"Expected lat shape {(trajectory.run_points,)}, but got {trajectory.pinpoint.lat.shape}"
        assert trajectory.pinpoint.lon.shape == (trajectory.run_points,), \
            f"Expected lon shape {(trajectory.run_points,)}, but got {trajectory.pinpoint.lon.shape}"
        assert trajectory.pinpoint.h_map.shape == (trajectory.run_points,), \
            f"Expected h_map shape {(trajectory.run_points,)}, but got {trajectory.pinpoint.h_map.shape}"