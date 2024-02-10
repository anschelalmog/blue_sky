import pytest
from unittest.mock import patch, MagicMock
from src.create_traj import CreateTraj

# Mock external dependencies
@pytest.fixture
def mock_mpd():
    with patch('src.utils.get_mpd', return_value=(1, 1)) as mock_func:
        yield mock_func

@pytest.fixture
def mock_map_data():
    map_data = MagicMock()
    map_data.ax_lat = [0, 1]  # Dummy values
    map_data.ax_lon = [0, 1]  # Dummy values
    map_data.grid = [[0, 1], [1, 2]]  # Dummy grid
    return map_data

# Example test: Ensure _create_pos correctly updates positions
def test_create_pos_updates_positions_correctly(mock_mpd, mock_map_data):
    # Setup
    args = MagicMock()
    args.init_lat = 0
    args.init_lon = 0
    args.init_height = 0
    args.avg_spd = 1
    args.acc_north = 0
    args.acc_east = 0
    args.acc_down = 0
    args.psi = 0
    args.theta = 0
    args.phi = 0
    args.psi_dot = 0
    args.theta_dot = 0
    args.phi_dot = 0
    args.time_vec = [0, 1]

    traj = CreateTraj(args)

    # Action
    traj._create_pos(mock_map_data)

    # Assert
    assert traj.pos.north[0] == args.init_lat
    assert traj.pos.east[0] == args.init_lon

# Add more tests for other methods like _create_euler, _create_vel, _create_traj, etc.
