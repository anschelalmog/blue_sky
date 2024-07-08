import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.pinpoint_calc import PinPoint
from src.create_traj import CreateTraj
from src.data_loaders import Map
from src.utils import cosd, sind


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

    @pytest.mark.parametrize("invalid_value", [np.inf, -600])
    def test_interpolation_error_handling(self, mock_args, loaded_map, invalid_value):
        trajectory = CreateTraj(mock_args)
        trajectory.create(loaded_map)
        h, w = loaded_map.grid.shape
        half_h, half_w = h // 2, w // 2

        # Introduce an invalid point that should cause interpolation to fail
        invalid_map_data = loaded_map
        invalid_map_data.grid = invalid_map_data.grid.astype(float)
        invalid_map_data.grid[half_h - 50:half_h + 50, half_w - 50:half_w + 50] = invalid_value
        with pytest.raises(ValueError):
            trajectory.pinpoint = PinPoint(trajectory.run_points).calc(trajectory, invalid_map_data)

    def test_boundary_conditions(self, mock_args, loaded_map, mock_traj):
        trajectory = mock_traj

        # Test lower boundary
        mock_traj.pos.lat[0] = loaded_map.axis['lat'][0] + 1e-5
        mock_traj.pos.lon[0] = loaded_map.axis['lon'][0] + 1e-5

        try:
            trajectory.pinpoint = PinPoint(trajectory.run_points).calc(trajectory, loaded_map)
        except Exception as e:
            assert False, f"Boundary condition test failed with exception: {e}"

    def test_position_calculation(self, mock_args):
        trajectory = CreateTraj(mock_args)
        trajectory._create_pos()

        init_north = mock_args.init_lat * trajectory.mpd_north
        init_east = mock_args.init_lon * trajectory.mpd_east

        expected_north_pos = init_north + trajectory.vel.north[
            0] * trajectory.time_vec + 0.5 * mock_args.acc_north * trajectory.time_vec ** 2
        expected_east_pos = init_east + trajectory.vel.east[
            0] * trajectory.time_vec + 0.5 * mock_args.acc_east * trajectory.time_vec ** 2
        expected_h_asl = mock_args.init_height + trajectory.vel.down[
            0] - 0.5 * mock_args.acc_down * trajectory.time_vec ** 2

        assert np.allclose(trajectory.pos.north, expected_north_pos, atol=1e-6), \
            "North position calculation is incorrect"
        assert np.allclose(trajectory.pos.east, expected_east_pos, atol=1e-6), \
            "East position calculation is incorrect"
        assert np.allclose(trajectory.pos.h_asl, expected_h_asl, atol=1e-6), \
            "Height ASL calculation is incorrect"

    def test_dcm_application(self, mock_args, loaded_map):
        trajectory = CreateTraj(mock_args)
        trajectory.create(loaded_map)

        initial_offsets = np.zeros((trajectory.run_points, 3))
        transformed_offsets = trajectory._apply_dcm(loaded_map)

        assert transformed_offsets.shape == initial_offsets.shape, \
            "DCM application did not produce the expected shape of transformed offsets"
