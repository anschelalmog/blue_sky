import pytest
import numpy as np
import os
from icecream import ic
from unittest.mock import patch, MagicMock, ANY
from src.data_loaders import Map, set_settings
from src.utils import get_mpd, cosd, sind, mocking_map


# ------------------------------------------------|
# must have args to create a map instance         |
# ------------------------------------------------|
# args.maps_dir,  args.map_level, args.map_res,   |
# args.init_lat, args.init_lon, args.avg_spd      |
# args.psi, args.theta,                           |
# args.acc_north, args.acc_east, args.time_end    |
# ------------------------------------------------|


class TestMap:
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

    @pytest.mark.parametrize(
        "lat, lon, avg_spd, psi, theta, time_end, expected_lat, expected_lon", [
            (31.5, 23.5, 250, 0, 0, 100, [31, 32], [23, 24]),
            (31.5, 23.5, 250, 0, 0, 1000, [31, 32], [23, 27]),
            (31.5, 23.5, 250, 45, 0, 1000, [31, 34], [23, 26]),
            (31.99, 23.5, 250, 90, 0, 100, [31, 33], [23, 24]),
            (31.5, 23.5, 1000, 90, 0, 100, [31, 33], [23, 24]),
            (31.1, 23.99, 250, 0, 0, 100, [31, 32], [23, 25]),
        ]
    )
    def test_set_map_boundaries(self, map_instance, lat, lon, avg_spd, psi, theta, time_end, expected_lat,
                                expected_lon):
        args = MagicMock(map_dir='Map', map_level=1, map_res=3,
                         init_lat=lat, init_lon=lon, avg_spd=avg_spd,
                         psi=psi, theta=theta, acc_north=0, acc_east=0, time_end=time_end)
        map_instance._set_map_boundaries(args)
        expected_bounds = {'lat': expected_lat, 'lon': expected_lon}
        assert map_instance.bounds == expected_bounds

    @pytest.mark.parametrize(
        "map_level, shape_, ext", [
            (1, (1201, 1201), 'mat'),
            (3, (3601, 3601), 'mat')
        ]
    )
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

    def test_map_save(self, map_instance):
        map_instance.grid = np.zeros((1201, 1201))
        map_instance.meta = {'some': 'meta'}
        map_instance.axis = {'some': 'axis'}
        map_instance.bounds = {'lat': [33, 34], 'lon': [45, 46]}
        map_instance.mpd = {'north': 1, 'east': 1}
        with patch('scipy.io.savemat') as mock_savemat:
            map_instance.save('path/to/save.mat')
            mock_savemat.assert_called_once_with('path/to/save.mat', ANY)


    def test_map_update(self, map_instance):
        original_map = map_instance
        original_map.bounds = {'lat': [37, 38], 'lon': [21, 22]}
        map_instance.meta = {'maps_dir': 'Map', 'tile_length': 1200, 'map_level': 1, 'ext': 'mat'}
        original_map._create_grid()
        original_map.update_map(38.4, 21.2)
        assert original_map.bounds['lat'][1] >= 35
        assert original_map.bounds['lon'][1] >= 22


