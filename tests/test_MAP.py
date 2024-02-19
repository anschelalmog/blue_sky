import pytest
import numpy as np
import os
from icecream import ic
from unittest.mock import patch, MagicMock
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


def test_map_initialization():
    map_instance = Map()
    assert map_instance.meta is None
    assert map_instance.axis is None
    assert map_instance.bounds is None
    assert map_instance.mpd is None
    assert map_instance.grid is None


def test_map_load():
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
    ])
def test_set_map_boundaries(lat, lon, avg_spd, psi, theta, time_end, expected_lat, expected_lon):
    args = MagicMock(map_dir='Map', map_level=1, map_res=3,
                     init_lat=lat, init_lon=lon, avg_spd=avg_spd,
                     psi=psi, theta=theta, acc_north=0, acc_east=0, time_end=time_end)
    map_instance = Map()
    map_instance._set_map_boundaries(args)
    expected_bounds = {'lat': expected_lat, 'lon': expected_lon}
    assert map_instance.bounds == expected_bounds, "Map boundaries did not match expected values."


@pytest.mark.parametrize(
    "map_level, shape_, ext", [
        (1, (1201, 1201), 'mat'),
        (3, (3601, 3601), 'mat')
    ])
def test_grid_creation(map_level, shape_, ext):
    map_instance = Map()
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
    expected_grid_shape = shape_
    assert map_instance.grid.shape == expected_grid_shape, "Grid shape did not match expected values."


@pytest.mark.parametrize("map_level, ext, expected_call", [
    (1, 'dt1', os.path.join(os.getcwd(), 'Map', 'Level1', 'MAP00', 'DTED', 'E045', 'n33.dt1')),
    (2, 'dt2', os.path.join(os.getcwd(), 'Map', 'Level2', 'MAP00', 'DTED', 'E045', 'n33.dt2')),
])
@patch('src.data_loaders.Map._load_tile')
def test_map_tile_loading_parametrized(mock_load_tile, map_level, ext, expected_call):
    mock_load_tile.return_value = np.zeros((1201, 1201))
    map_instance = Map()
    map_instance.meta = {'maps_dir': 'Map', 'tile_length': 1200, 'map_level': map_level, 'ext': ext}
    map_instance.bounds = {'lat': [33, 34], 'lon': [45, 46]}
    map_instance._create_grid()
    mock_load_tile.assert_called_with(expected_call, 1200)


def test_map_update():
    original_map = Map()
    original_map.bounds = {'lat': [33, 34], 'lon': [45, 46]}
    original_map._create_grid()
    original_map.update_map(35, 47)  # New coordinates that extend the map boundaries
    assert original_map.bounds['lat'][1] >= 35, "Map did not update northern boundary as expected."
    assert original_map.bounds['lon'][1] >= 47, "Map did not update eastern boundary as expected."


def test_map_save():
    map_instance = Map()
    map_instance.grid = np.zeros((1201, 1201))
    map_instance.meta = {'some': 'meta'}
    map_instance.axis = {'some': 'axis'}
    map_instance.bounds = {'lat': [33, 34], 'lon': [45, 46]}
    map_instance.mpd = {'north': 1, 'east': 1}
    with patch('scipy.io.savemat') as mock_savemat:
        map_instance.save('path/to/save.mat')
        mock_savemat.assert_called_once()
        args, kwargs = mock_savemat.call_args
        assert 'path/to/save.mat' in args, "Map save path did not match."
