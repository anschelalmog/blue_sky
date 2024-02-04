import pytest
from src.create_traj import CreateTraj
from src.data_loaders import Map, set_settings
@pytest.fixture
def setup_create_traj():
    args = set_settings()
    map_data = Map(args).load()
    create_traj = CreateTraj(args)
    return create_traj, map_data

def test_euler_creation(setup_create_traj):
    create_traj, _ = setup_create_traj
    create_traj._create_euler()  # Directly calling the protected method for testing
    assert create_traj.euler.psi[0] == create_traj.args.psi  # Initial condition check
    # Add more assertions as needed

def test_velocity_creation(setup_create_traj):
    create_traj, _ = setup_create_traj
    create_traj._create_vel()
    # Assertions to verify velocities are correctly calculated

def test_position_creation(setup_create_traj):
    create_traj, map_data = setup_create_traj
    create_traj._create_pos
