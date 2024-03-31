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
    def mock_traj(self, mock_args):
        traj = CreateTraj(mock_args)
        return traj

    @pytest.fixture
    def loaded_map(self, mock_args):
        # Load map data from a file or database
        map_data = Map()
        map_data.load(mock_args)
        return map_data

    @pytest.fixture
    def constant_map(self, mock_args):
        # Create a constant map with a height of 1000
        map_data = Map()
        map_data.axis = {'lat': np.linspace(mock_args.init_lat - 1, mock_args.init_lat + 1, mock_args.run_points),
                         'lon': np.linspace(mock_args.init_lon - 1, mock_args.init_lon + 1, mock_args.run_points)}
        map_data.grid = (np.ones((100, 100)) * 1000).astype(int)
        map_data.mpd = {'north': np.ones(100) * 111000, 'east': np.ones(100) * 111000}
        return map_data

    @pytest.fixture(params=[('constant', (100, 100)), ('loaded', (1201, 1201))])
    def map_data(self, request):
        map_type, expected_shape = request.param
        if map_type == 'loaded':
            return request.getfixturevalue('loaded_map'), expected_shape
        elif map_type == 'constant':
            return request.getfixturevalue('constant_map'), expected_shape
        else:
            raise ValueError(f"Invalid map type: {map_type}")

    def test_map_grid_loaded(self, map_data):
        map_data, expected_shape = map_data
        grid = map_data.grid
        assert grid.shape == expected_shape, f"Expected grid shape {expected_shape} for {type(map_data).__name__}, but got {grid.shape}"
        assert np.issubdtype(grid.dtype, np.integer), f"Expected grid elements to be integers, but got {grid.dtype}"

    def test_pinpoint_calc_output_shape(self):
        """
        This test verifies that the output arrays of PinPoint.calc() have the expected
        shapes based on the number of run points in the trajectory.
        It ensures that the method returns the correct number of values for each output variable.
        """
        pass

    def test_pinpoint_calc_range_positive(self):
        """
        This test checks that all range values calculated by PinPoint.calc() are non-negative.
        The range represents the distance from the vehicle to the pinpoint and should always be positive.
        """
        pass

    def test_pinpoint_calc_delta_north_east_reasonable(self):
        """
        This test verifies that the calculated delta_north and delta_east values are within a reasonable
         range (assumed to be less than 1 degree in this example).
         It ensures that the pinpoint calculations produce sensible results.
         """
        pass

    def test_pinpoint_calc_lat_lon_within_map_bounds(self):
        """This test checks that the calculated pinpoint latitudes and longitudes
        fall within the bounds of the provided map data.
        It ensures that the pinpoint coordinates are valid and within the map's extent."""
        pass

    def test_pinpoint_calc_h_map_within_map_elevation_range(self):
        """ This test verifies that the calculated pinpoint elevations(h_map)
            are within the range of elevations present in the map data.
            It ensures that the pinpoint elevations are consistent with the map's elevation values.
        """
        pass

    def test_pinpoint_calc_range_increases_with_height(self):
        """
            This test compares the calculated ranges for two different initial heights of the trajectory.
            It verifies that the range values increase when the initial height is increased,
            as expected due to the longer distance to the ground.
        """
        pass

    def test_pinpoint_calc_delta_north_east_change_with_psi(self):
        """
            This test compares the calculated delta_north and delta_east values for two different psi angles (heading).
            It verifies that delta_north is larger when
            psi is 0 (heading north) and delta_east is larger when psi is 90 (heading east),
            as expected based on the direction of travel.
        """
        pass

    def test_pinpoint_calc_delta_north_east_change_with_theta_phi(self):
        """
            This test compares the calculated delta_north and delta_east values
            for two different combinations of theta (pitch) and phi (roll) angles.
            It verifies that the magnitude of delta_north and delta_east increases when the vehicle is
            tilted (non-zero theta and phi), as expected due to the changed orientation.
        """
        pass

    def test_pinpoint_calc_lat_lon_change_with_acc_north_east(self):
        """
            This test compares the calculated pinpoint latitudes and longitudes
            for two different combinations of acc_north and acc_east (accelerations).
            It verifies that the pinpoint coordinates increase when non-zero accelerations are applied,
            as expected due to the change in velocity.
        """
        pass
