import numpy as np
from ..core.base import BaseTraj
import matplotlib.pyplot as plt


class IMUErrors:
    """
    A class to manage and store error parameters for various IMU sensors.

    This class provides a structured way to define, store, and retrieve
    error parameters for different sensors in an inertial measurement unit.

    Attributes:
        imu_errors (dict): Dictionary of error parameters for IMU sensors
    """

    def __init__(self, imu_errors=None, set_defaults=True):
        """
        Initialize the IMUErrors class.

        Args:
            imu_errors (dict, optional): Dictionary to initialize IMU errors
            set_defaults (bool, optional): Whether to set default errors
        """
        self.imu_errors = {} if imu_errors is None else imu_errors
        if set_defaults:
            self.create_defaults_errors()
            self.print_table()

    def set_imu_error(self, imu_name, amplitude=0, drift=0, bias=0):
        """
        Set error parameters for a specific IMU sensor.

        Args:
            imu_name (str): Name of the IMU sensor
            amplitude (float, optional): Amplitude error
            drift (float, optional): Drift error
            bias (float, optional): Bias error
        """
        self.imu_errors[imu_name] = {'amplitude': amplitude, 'drift': drift, 'bias': bias}

    def get_imu_error(self, imu_name):
        """
        Get error parameters for a specific IMU sensor.

        Args:
            imu_name (str): Name of the IMU sensor

        Returns:
            dict: Error parameters for the specified sensor
        """
        return self.imu_errors.get(imu_name, {'amplitude': 0, 'drift': 0, 'bias': 0})

    def create_defaults_errors(self):
        """
        Create and set default error parameters for predefined IMU sensors.
        """
        # Define default errors for various sensors with units in comments
        self.set_imu_error('altimeter', amplitude=0, drift=0, bias=0)        # [m   m/s    m  ]
        self.set_imu_error('barometer', amplitude=0, drift=0, bias=0)        # [m   m/s    m  ]
        self.set_imu_error('gyroscope', amplitude=0, drift=0, bias=0)        # [deg deg/s  deg]
        self.set_imu_error('accelerometer', amplitude=0, drift=0, bias=0)    # [m/s^2 m/s^3 m/s^2]
        self.set_imu_error('velocity meter', amplitude=0, drift=0, bias=0)   # [m/s  m/s^2  m/s]
        self.set_imu_error('position', amplitude=0, drift=0, bias=0)         # [m  m/s^2 m]

    def print_table(self):
        """
        Print a formatted table of the IMU error parameters.
        """
        print("=" * 65)
        print("IMU Errors".center(65))
        print("=" * 65)
        print("| {:^20} | {:^10} | {:^10} | {:^10} |".format("IMU Name", "Amplitude", "Drift", "Bias"))
        print("-" * 65)
        for imu_name, error in self.imu_errors.items():
            print("| {:^20} | {:^10} | {:^10} | {:^10} |".format(
                imu_name, error['amplitude'], error['drift'], error['bias']))
        print("-" * 65)


class RunErrors(BaseTraj):
    def __init__(self, used_traj, est_traj, covars):
        super().__init__(used_traj.run_points)
        self.pos.north = (used_traj.pos.lat - est_traj.pos.lat)
        self.pos.east = (used_traj.pos.lon - est_traj.pos.lon)  # used_traj.mpd_east
        self.pos.h_asl = used_traj.pos.h_asl - est_traj.pos.h_asl
        #
        self.vel.north = (used_traj.vel.north - est_traj.vel.north)
        self.vel.east = (used_traj.vel.east - est_traj.vel.east)
        self.vel.down = (used_traj.vel.down - est_traj.vel.down)
        #
        self.euler.psi = used_traj.euler.psi - est_traj.euler.psi
        self.euler.theta = used_traj.euler.theta - est_traj.euler.theta
        self.euler.phi = used_traj.euler.phi - est_traj.euler.phi
        #

        self.metrics = {
            'pos': {
                'north': {
                    'rmse': np.sqrt(np.mean(self.pos.north ** 2)),
                    'max_abs_error': np.max(np.abs(self.pos.north)),
                    'error_bound_percentage': np.mean(np.abs(self.pos.north) <= np.abs(covars.pos.north)) * 100
                },
                'east': {
                    'rmse': np.sqrt(np.mean(self.pos.east ** 2)),
                    'max_abs_error': np.max(np.abs(self.pos.east)),
                    'error_bound_percentage': np.mean(np.abs(self.pos.east) <= np.abs(covars.pos.east)) * 100
                },
                'h_asl': {
                    'rmse': np.sqrt(np.mean(self.pos.h_asl ** 2)),
                    'max_abs_error': np.max(np.abs(self.pos.h_asl)),
                    'error_bound_percentage': np.mean(np.abs(self.pos.h_asl) <= np.abs(covars.pos.h_asl)) * 100
                }
            },
            'vel': {
                'north': {
                    'rmse': np.sqrt(np.mean(self.vel.north ** 2)),
                    'max_abs_error': np.max(np.abs(self.vel.north)),
                    'error_bound_percentage': np.mean(np.abs(self.vel.north) <= np.abs(covars.vel.north)) * 100
                },
                'east': {
                    'rmse': np.sqrt(np.mean(self.vel.east ** 2)),
                    'max_abs_error': np.max(np.abs(self.vel.east)),
                    'error_bound_percentage': np.mean(np.abs(self.vel.east) <= np.abs(covars.vel.east)) * 100
                },
                'down': {
                    'rmse': np.sqrt(np.mean(self.vel.down ** 2)),
                    'max_abs_error': np.max(np.abs(self.vel.down)),
                    'error_bound_percentage': np.mean(np.abs(self.vel.down) <= np.abs(covars.vel.down)) * 100
                }
            },
            'euler': {
                'psi': {
                    'rmse': np.sqrt(np.mean(self.euler.psi ** 2)),
                    'max_abs_error': np.max(np.abs(self.euler.psi)),
                    'error_bound_percentage': np.mean(np.abs(self.euler.psi) <= np.abs(covars.euler.psi)) * 100
                },
                'theta': {
                    'rmse': np.sqrt(np.mean(self.euler.theta ** 2)),
                    'max_abs_error': np.max(np.abs(self.euler.theta)),
                    'error_bound_percentage': np.mean(np.abs(self.euler.theta) <= np.abs(covars.euler.theta)) * 100
                },
                'phi': {
                    'rmse': np.sqrt(np.mean(self.euler.phi ** 2)),
                    'max_abs_error': np.max(np.abs(self.euler.phi)),
                    'error_bound_percentage': np.mean(np.abs(self.euler.phi) <= np.abs(covars.euler.phi)) * 100
                }
            }
        }


class Covariances(BaseTraj):
    def __init__(self, covariances, traj):
        super().__init__(covariances.shape[2])
        #
        self.pos.north = np.sqrt(covariances[0, 0, :])
        self.pos.east = np.sqrt(covariances[1, 1, :])
        self.pos.h_asl = np.sqrt(covariances[2, 2, :])
        #
        self.vel.north = np.sqrt(covariances[3, 3, :])
        self.vel.east = np.sqrt(covariances[4, 4, :])
        self.vel.down = np.sqrt(covariances[5, 5, :])
        #
        self.acc.north = np.sqrt(covariances[6, 6, :])
        self.acc.north = np.sqrt(covariances[6, 6, :])
        self.acc.east = np.sqrt(covariances[7, 7, :])
        self.acc.down = np.sqrt(covariances[8, 8, :])
        #
        self.euler.psi = np.sqrt(covariances[9, 9, :])
        self.euler.theta = np.sqrt(covariances[10, 10, :])
        self.euler.phi = np.sqrt(covariances[11, 11, :])

        attr_indices = {
            'pos': {'north': 0, 'east': 1, 'h_asl': 2},
            'vel': {'north': 3, 'east': 4, 'down': 5},
            'acc': {'north': 6, 'east': 7, 'down': 8},
            'euler': {'psi': 9, 'theta': 10, 'phi': 11}
        }

        # Iterate over the attribute groups and their items
        for attr_group, indices in attr_indices.items():
            for attr_name, idx in indices.items():
                # Use sqrt to calculate the standard deviation from the variance
                setattr(getattr(self, attr_group), attr_name, np.sqrt(covariances[idx, idx, :]))


def calc_errors_covariances(meas_traj, estimation_results):
    covariances = Covariances(estimation_results.params.P_est, estimation_results.traj)
    errors = RunErrors(meas_traj, estimation_results.traj, covariances)
    return errors, covariances