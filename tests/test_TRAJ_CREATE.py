import pytest
from unittest.mock import Mock
import unittest
import numpy as np
from icecream import ic
from scipy.interpolate import RegularGridInterpolator

from BLUE_SKY.create_traj import CreateTraj
from BLUE_SKY.utils import get_mpd
from BLUE_SKY.data_loaders import set_settings


class TestCreateTraj(unittest.TestCase):
    """
    Test suite for the CreateTraj class.

    This class contains a series of unit tests designed to verify the functionality and reliability of the CreateTraj
    class. It tests the initialization of a CreateTraj instance, the behavior of Euler angles, acceleration,
    velocity, and position calculations, as well as the trajectory creation with respect to non-uniform grid handling.

    Fixtures:
     - mock_args: Provides a mocked argument object with predefined attributes necessary for initializing a
                  CreateTraj instance.
     - mock_map_data: Provides mocked map data including axis information and a grid representing
                      elevation data used in trajectory creation.

    Methods:
    - test_initialization:
        Verifies that a new CreateTraj instance is correctly initialized with attributes
        from the provided mock arguments.
    - test_euler_angles:
        Checks if Euler angles (psi, theta, phi) are approximated well by a polynomial of a given degree,
        ensuring the rotational motion is represented accurately.
    - test_acc_behavior:
        Validates the acceleration behavior in the north, east, and down directions, ensuring it can
        be approximated by a polynomial of a specified degree, indicating predictable and consistent acceleration
        patterns.
    - test_velocity_behavior:
        Confirms that the velocity components in the north, east, and down directions
        are well represented by a polynomial of a given degree, reflecting realistic velocity changes over time.
    - test_pos_behavior:
        Examines the position calculation in the north, east, and vertical (h_asl) directions under
        different acceleration scenarios, ensuring the position changes are consistent with the applied
        accelerations and can be approximated by an appropriate polynomial.
    - test_create_traj_non_uniform_grid_handling:
        Tests the CreateTraj class's ability to handle non-uniform grid data during trajectory creation,
        ensuring each trajectory point is assigned a correct height value from the map data,
        even when latitude and longitude grids are not uniform.
    """
    @pytest.fixture
    def mock_args(self):
        # Mocking the args object with necessary attributes for CreateTraj initialization
        args = Mock()
        args.run_points = 100
        args.time_vec = np.linspace(0, 99, 100)
        args.init_lat = 37.5
        args.init_lon = 21.5
        args.init_height = 5000
        args.avg_spd = 250
        args.acc_north = 0
        args.acc_east = 0
        args.acc_down = 0
        args.psi = -45
        args.theta = 22
        args.phi = 12
        args.psi_dot = 1
        args.theta_dot = 3
        args.phi_dot = 4
        return args

    @pytest.fixture
    def mock_map_data(self):
        # Mocking the map_data required for _create_traj method
        map_data = Mock()
        map_data.axis = {'lat': np.linspace(37, 38, 101), 'lon': np.linspace(21, 22, 101)}
        map_data.grid = np.random.rand(101, 101) * 100  # Simulate random elevation data
        return map_data

    def test_initialization(self, mock_args):
        traj = CreateTraj(mock_args)
        assert traj.run_points == mock_args.run_points
        assert np.array_equal(traj.time_vec, mock_args.time_vec)
        assert traj.inits['lat'] == mock_args.init_lat
        assert traj.inits['lat'] == mock_args.init_lat

    @pytest.mark.parametrize("angle", ['psi', 'theta', 'phi'])
    @pytest.mark.parametrize("degree,expected_rss_threshold", [(1, 1e-10), (2, 1)])
    def test_euler_angles(self, mock_args, angle, degree, expected_rss_threshold):
        traj = CreateTraj(mock_args)
        traj._create_euler()

        y = getattr(traj.euler, angle)
        x = traj.time_vec

        # Fit a polynom
        poly_coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(poly_coeffs, x)

        # Calculate RSS
        rss = np.sum((y - y_pred) ** 2)
        ic(degree, rss)

        assert rss < expected_rss_threshold, f"Euler angle {angle} is not well approximated by a degree {degree}"

    @pytest.mark.parametrize("acc_component, expected_data_attr",
                             [('north', 'acc_north'), ('east', 'acc_east'), ('down', 'acc_down')])
    @pytest.mark.parametrize("degree,expected_rss_threshold", [(1, 1), (2, 10)])
    def test_acc_behavior(self, mock_args, acc_component, expected_data_attr, degree, expected_rss_threshold):
        traj = CreateTraj(mock_args)
        traj._create_acc()

        y = getattr(traj.acc, acc_component)
        x = np.arange(len(y))

        # Fit a polynomial
        poly_coeff = np.polyfit(x, y, degree)
        y_pred = np.polyval(poly_coeff, x)

        # Calculate RSS
        rss = np.sum((y - y_pred) ** 2)
        ic(degree, rss)

        assert rss < expected_rss_threshold, (f"Acceleration component {acc_component} is not well"
                                              f" approximated by a degree {degree} polynomial")

    @pytest.mark.parametrize("velocity_component", ['north', 'east', 'down'])
    @pytest.mark.parametrize("degree,expected_rss_threshold", [(1, 1e-10), (2, 1)])
    def test_velocity_behavior(self, mock_args, velocity_component, degree, expected_rss_threshold):
        traj = CreateTraj(mock_args)
        traj._create_vel()

        y = getattr(traj.vel, velocity_component)
        x = traj.time_vec

        # Fit a polynomial
        poly_coeff = np.polyfit(x, y, degree)
        y_pred = np.polyval(poly_coeff, x)

        # Calculate RSS
        rss = np.sum((y - y_pred) ** 2)
        ic(degree, rss)

        assert rss < expected_rss_threshold, (f"Velocity component {velocity_component} "
                                              f"is not well approximated by a degree {degree}")

    @pytest.mark.parametrize("axis", ['north', 'east', 'h_asl'])
    @pytest.mark.parametrize("acc_values", [(-1, 0, 0), (0, 2, 0), (0, 0, 3)])
    def test_pos_behavior(self, mock_args, axis, acc_values):
        mock_args.acc_north, mock_args.acc_east, mock_args.acc_down = acc_values
        traj = CreateTraj(mock_args)
        traj._create_acc()
        traj._create_pos()

        # Get the acceleration and position for the current axis
        acc = getattr(traj.acc, axis if axis in ['north', 'east'] else 'down')  # Use 'down' for vertical acceleration
        pos = getattr(traj.pos, axis)
        x = np.arange(len(pos))

        # Determine if acceleration is almost constant

        if np.allclose(acc, 0, atol=1e-08):  # Threshold for considering acceleration as constant
            degree = 1
            expected_rss_threshold = 1
        else:
            degree = 2
            expected_rss_threshold = 10

        # Fit a polynomial of determined degree
        poly_coeff = np.polyfit(x, pos, degree)
        pos_pred = np.polyval(poly_coeff, x)

        # Calculate RSS
        rss = np.sum((pos - pos_pred) ** 2)

        assert rss < expected_rss_threshold, (
            f"Position component {axis} with acceleration values {acc_values} "
            f"is not well approximated by a degree {degree} polynomial; RSS: {rss}")

    def test_create_traj_non_uniform_grid_handling(self, mock_args, mock_map_data):
        mock_map_data.axis['lat'] = np.linspace(37, 38, 50)
        mock_map_data.axis['lon'] = np.linspace(21, 22, 150)
        mock_map_data.grid = np.random.rand(50, 150) * 100

        traj = CreateTraj(mock_args)
        traj.pos.lat = np.linspace(37, 38, mock_args.run_points)
        traj.pos.lon = np.linspace(21, 22, mock_args.run_points)

        # Execute the method under test
        traj._create_traj(mock_map_data)

        assert len(traj.pos.h_map) == mock_args.run_points  # Ensure we have a height for each trajectory point
        assert np.all(traj.pos.h_map >= 0)  # Assuming map data represents heights, they should be non-negative
