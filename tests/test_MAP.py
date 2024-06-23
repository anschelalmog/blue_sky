import pytest
import numpy as np
import tempfile
from unittest.mock import patch, MagicMock, ANY
from src.data_loaders import Map

""" 
|-------------------------------------------------|
| must have args to create a full map instance    |
|-------------------------------------------------|
| args.maps_dir,  args.map_level, args.map_res,   |
| args.init_lat, args.init_lon, args.avg_spd      |
| args.psi, args.theta,                           |
| args.acc_north, args.acc_east, args.time_end    |
|-------------------------------------------------|
"""


class TestMap:
    """
    Test suite for the Map class.

    This class contains a series of unit tests designed to verify the functionality and reliability of the Map class.
    It tests the initial state of a new Map instance, the loading of map data, boundary setting based on provided
    parameters, grid creation, tile loading, and the ability to update map boundaries and save the map state.

    Fixtures:
    - map_instance: Provides a fresh instance of the Map class for each test method. - mock_load_tile:
    Mocks the _load_tile method of the Map class to return a predefined grid instead of reading from files.

    Methods:
    - test_initialization: Verifies that a new Map instance has all its attributes set to None initially.
    - test_map_load: Checks if the Map instance correctly initializes its attributes when the load method is called
                     with simulated arguments.
    - test_set_map_boundaries: Ensures that the map boundaries are correctly calculated and set based on various
                               simulated flight parameters.
    - test_grid_creation: Validates the creation of the map grid with the expected
                          shape based on the map's level of detail.
    - test_load_tile_and_grid_state: Tests the _load_tile method's response to existing and non-existing
                                    files and the resulting state of the map grid.
    - test_map_save: Confirms that the Map instance can successfully save its current state to a file.
    - test_update_map: Examines the update_map method's ability to expand
                       the map's boundaries when new coordinates are introduced.
    """

    @pytest.fixture
    def map_instance(self):
        return Map()

    @pytest.fixture
    def mock_load_tile(self, monkeypatch):
        def mock(*args, **kwargs):
            return np.zeros((1201, 1201))

        monkeypatch.setattr(Map, "_load_tile", mock)

    def test_initialization(self, map_instance):
        assert map_instance.meta is None
        assert map_instance.axis is None
        assert map_instance.bounds is None
        assert map_instance.mpd is None
        assert map_instance.grid is None

    def test_map_load(self, map_instance):
        # Create a MagicMock object to simulate the args
        args = MagicMock(map_dir='Map', map_level=1, map_res=3,
                         init_lat=31.5, init_lon=23.5, avg_spd=250,
                         psi=0, theta=0, phi=0,
                         acc_north=0, acc_east=0, time_end=100)

        map_instance = Map().load(args)

        assert map_instance.meta is not None, "Expected 'meta' to be initialized but found None."
        assert map_instance.axis is not None, "Expected 'axis' to be initialized but found None."
        assert map_instance.bounds is not None, "Expected 'bounds' to be initialized but found None."
        assert map_instance.mpd is not None, "Expected 'mpd' to be initialized but found None."
        assert map_instance.grid is not None, "Expected 'grid' to be initialized but found None."

    def test_map_save(self, map_instance):
        map_instance.grid = np.zeros((1201, 1201))
        map_instance.meta = {'some': 'meta'}
        map_instance.axis = {'some': 'axis'}
        map_instance.bounds = {'lat': [33, 34], 'lon': [45, 46]}
        map_instance.mpd = {'north': 1, 'east': 1}

        with tempfile.NamedTemporaryFile(suffix='.mat') as temp_file:
            file_path = temp_file.name
            with patch('scipy.io.savemat') as mock_savemat:
                map_instance.save(file_path)
                mock_savemat.assert_called_once_with(file_path, ANY)

    @pytest.mark.parametrize("lat, lon, avg_spd, psi, theta, time_end, expected_lat, expected_lon", [
        (31.5, 23.5, 250, 0, 0, 100, [31, 32], [23, 24]),
        (31.5, 23.5, 250, 0, 0, 1000, [31, 32], [23, 27]),
        (31.5, 23.5, 250, 45, 0, 1000, [31, 34], [23, 26]),
        (31.99, 23.5, 250, 90, 0, 100, [31, 33], [23, 24]),
        (31.5, 23.5, 1000, 90, 0, 100, [31, 33], [23, 24]),
        (31.1, 23.99, 250, 0, 0, 100, [31, 32], [23, 25]),
    ])
    def test_set_map_boundaries(self, map_instance, lat, lon, avg_spd, psi, theta, time_end, expected_lat,
                                expected_lon):
        args = MagicMock(map_dir='Map', map_level=1, map_res=3,
                         init_lat=lat, init_lon=lon, avg_spd=avg_spd,
                         psi=psi, theta=theta, acc_north=0, acc_east=0, time_end=time_end)
        map_instance._set_map_boundaries(args)
        expected_bounds = {'lat': expected_lat, 'lon': expected_lon}
        assert map_instance.bounds == expected_bounds

    @pytest.mark.parametrize("map_level, shape_, ext", [
        (1, (1201, 1201), 'mat'),
        (3, (3601, 3601), 'mat')
    ])
    def test_grid_creation(self, map_instance, map_level, shape_, ext):
        map_instance.bounds = {'lat': [33, 34], 'lon': [45, 46]}
        map_res = 1 if map_level == 3 else 3
        tile_length, map_level = (1200, 1) if map_res == 3 else (3600, 3)
        map_instance.meta = {
            'maps_dir': 'Map',
            'map_res': map_res,
            'map_level': map_level,
            'rate': map_res / 3600,
            'tile_length': tile_length,
            'ext': ext
        }
        map_instance._create_grid()
        assert map_instance.grid.shape == shape_, "Grid shape did not match expected values."

    @pytest.mark.parametrize("exists, expected_result", [
        (True, True),  # File exists, expect True
        (False, False)  # File doesn't exist, expect False
    ])
    def test_load_tile_and_grid_state(self, map_instance, monkeypatch, exists, expected_result):
        monkeypatch.setattr("os.path.exists", lambda path: exists)

        def mock_load_tile(*args, **kwargs):
            return np.zeros((1201, 1201)) if exists else None

        monkeypatch.setattr(map_instance, "_load_tile", mock_load_tile)
        map_instance.meta = {'maps_dir': 'Map', 'tile_length': 1200, 'map_level': 1, 'ext': 'mat'}
        map_instance.bounds = {'lat': [33, 34], 'lon': [45, 46]}

        map_instance._create_grid()

        assert (map_instance._load_tile() is not None) == expected_result, \
            f"Expected _load_tile to return {'a non-None value' if expected_result else 'None'}"

        assert map_instance.grid is not None, "Expected grid to be created even if the file doesn't exist"

    @pytest.mark.parametrize("new_lat, new_lon, expected_update", [
        (40.1, 22, True),  # North
        (40.1, 23.1, True),  # North-East
        (39, 23.1, True),  # East
        (37.9, 23.1, True),  # South-East
        (37.9, 22, True),  # South
        (37.9, 20.9, True),  # South-West
        (39, 20.9, True),  # West
        (40.1, 20.9, True),  # North-West
        (38, 22, False),  # Within existing boundaries, no update expected
    ])
    def test_update_map(self, map_instance, new_lat, new_lon, expected_update, mock_load_tile):
        """
           Tests the `update_map` method's ability to correctly update the map boundaries when new coordinates
           are introduced that lie outside the current map boundaries.

           The method is tested for all eight neighboring directions around the existing map grid:
           North, North-East, East, South-East, South, South-West, West, and North-West. Additionally, a case
           where the new coordinates lie within the existing boundaries is tested to ensure no unnecessary updates
           are made.

           Parameters:
           - new_lat (float): The latitude of the new point to be added to the map.
           - new_lon (float): The longitude of the new point to be added to the map.
           - expected_update (bool): A flag indicating whether an update to the map boundaries is expected.

           Expected Results:
           - If `expected_update` is True, the map's latitude and longitude boundaries should be updated to
             include the new coordinates. This is verified by checking that the new coordinates fall within
             the updated boundaries of the map.
           - If `expected_update` is False (indicating the new point lies within the current boundaries),
             the map's boundaries should remain unchanged. This case tests the method's ability to avoid
             unnecessary updates when the map already encompasses the new coordinates.
           """
        # Mocking _create_grid and _set_axis to isolate update_map functionality
        with patch.object(Map, '_create_grid'), patch.object(Map, '_set_axis'):
            map_instance.bounds = {'lat': [37, 40], 'lon': [20, 23]}  # Setting initial boundaries
            map_instance.update_map(new_lat, new_lon)

            # Check if the update_map method attempts to update the map
            if expected_update:
                assert map_instance.bounds['lat'][0] <= new_lat <= map_instance.bounds['lat'][
                    1], "Latitude bounds were not updated correctly."
                assert map_instance.bounds['lon'][0] <= new_lon <= map_instance.bounds['lon'][
                    1], "Longitude bounds were not updated correctly."
            else:
                # If no update is expected, the bounds should remain unchanged
                assert map_instance.bounds == {'lat': [37, 40],
                                               'lon': [20, 23]}, "Map boundaries should not be updated."
