import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.interpolate import RegularGridInterpolator

from .base import BaseKalmanFilter
from ..utils.math import cosd
from ..utils.progress import progress_bar


class UKFParams:
    """
    Parameters for FilterPy Unscented Kalman Filter.

    This class stores all matrices and parameters needed for UKF estimation.

    Attributes:
        state_size: Size of the state vector
        run_points: Number of trajectory points
        P_est: Error covariance matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        Rc: Range-dependent component of R
        Rfit: Fit error component of R
        dX: State correction
        SN: North slope
        SE: East slope
    """
    def __init__(self, ukf):
        """
        Initialize UKF parameters.

        Args:
            ukf: Unscented Kalman Filter instance containing configuration
        """
        self.state_size = ukf.state_size
        self.run_points = ukf.run_points
        self.P_est = np.zeros((self.state_size, self.state_size, self.run_points))
        self.Q = np.zeros((self.state_size, self.state_size))
        self.R = np.zeros(self.run_points)
        self.Rc = np.zeros(self.run_points)
        self.Rfit = np.zeros(self.run_points)
        self.dX = np.zeros((self.state_size, self.run_points))
        self.SN = np.zeros(self.run_points)
        self.SE = np.zeros(self.run_points)
        self.K = np.zeros((self.state_size, self.run_points))
        self.Z = np.zeros(self.run_points)


class FilterPyUKF(BaseKalmanFilter):
    """
    FilterPy Unscented Kalman Filter implementation.

    This class adapts FilterPy's UKF to the existing codebase structure,
    maintaining the same interface while using FilterPy for the core computations.

    Attributes:
        params (UKFParams): Filter parameters
        ukf (UnscentedKalmanFilter): The FilterPy UKF instance
        map_data: Map data for terrain information
    """
    def __init__(self, args):
        """
        Initialize the FilterPy UKF filter.

        Args:
            args: Configuration arguments
        """
        super().__init__(args)
        self.params = None  # Will be initialized in _initialize_params
        self.ukf = None     # FilterPy's UnscentedKalmanFilter instance
        self.map_data = None  # For terrain data access in state and measurement functions
        self.current_measurement = None  # To store current measurement for use in state transition function

    def run(self, map_data, meas):
        """
        Run the Unscented Kalman Filter to estimate trajectory.

        This method adapts the FilterPy UKF to the existing framework.

        Args:
            map_data: Map containing terrain information
            meas: Measurements to use for estimation

        Returns:
            self: Updated instance with estimation results
        """
        self._initialize_traj(meas)
        self._initialize_params()
        self.map_data = map_data
        self.current_measurement = meas  # Store measurements for state transition function

        # Define state transition and measurement functions for FilterPy UKF
        def fx(x, dt):
            """State transition function"""
            # Extract current state
            pos_lat, pos_lon, h_asl = x[0], x[1], x[2]
            vel_north, vel_east, vel_down = x[3], x[4], x[5]
            acc_north, acc_east, acc_down = x[6], x[7], x[8]
            psi, theta, phi = x[9], x[10], x[11]

            # Current time step
            i = self.curr_state

            # Apply state transition model (kinematic equations)
            new_x = x.copy()

            # Update positions based on velocity and acceleration
            new_x[0] = pos_lat + (vel_north * dt + 0.5 * self.current_measurement.acc.north[i-1] * dt**2) / self.current_measurement.mpd_north[i-1]
            new_x[1] = pos_lon + (vel_east * dt + 0.5 * self.current_measurement.acc.east[i-1] * dt**2) / self.current_measurement.mpd_east[i-1]
            new_x[2] = h_asl + vel_down * dt + 0.5 * self.current_measurement.acc.down[i-1] * dt**2

            # Update velocities based on acceleration
            new_x[3] = vel_north + self.current_measurement.acc.north[i-1] * dt
            new_x[4] = vel_east + self.current_measurement.acc.east[i-1] * dt
            new_x[5] = vel_down + self.current_measurement.acc.down[i-1] * dt

            # Apply linear change to Euler angles
            new_x[9] = psi + ((self.current_measurement.euler.psi[i] - self.current_measurement.euler.psi[i-1]) * dt)
            new_x[10] = theta + ((self.current_measurement.euler.theta[i] - self.current_measurement.euler.theta[i-1]) * dt)
            new_x[11] = phi + ((self.current_measurement.euler.phi[i] - self.current_measurement.euler.phi[i-1]) * dt)

            return new_x

        def hx(x):
            """Measurement function - returns predicted measurement"""
            # Extract state components
            pos_lat, pos_lon, h_asl = x[0], x[1], x[2]

            # Current time step
            i = self.curr_state

            # Calculate height above ground level based on current state
            try:
                # Get map height at predicted position
                h_map = RegularGridInterpolator(
                    (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)(
                    (pos_lat, pos_lon))

                # Use current measured h_agl for simplicity
                jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
                h_agl = meas.pinpoint.range[i] * jac

                # Predicted measurement is height error
                return np.array([h_asl - h_agl - h_map])
            except ValueError:
                # Fallback if interpolation fails
                return np.array([0.0])

        # Create sigma points
        sigma_points = MerweScaledSigmaPoints(
            n=self.state_size,
            alpha=1e-3,  # Small alpha for tightly packed sigma points
            beta=2.0,    # Optimal for Gaussian distributions
            kappa=0.0    # Default value
        )

        # Create and initialize FilterPy UKF
        self.ukf = UnscentedKalmanFilter(
            dim_x=self.state_size,
            dim_z=1,  # One measurement (height error)
            dt=self.del_t,
            fx=fx,
            hx=hx,
            points=sigma_points
        )

        # Set initial state and covariance
        self.ukf.x = self._get_initial_state_vector(meas)
        self.ukf.P = self.params.P_est[:, :, 0]
        self.ukf.Q = self.params.Q

        desc = "Estimating Trajectory with FilterPy UKF"
        for i in progress_bar(range(1, self.run_points), desc):
            self.curr_state = i

            try:
                # Get measurement at current time
                lat, lon = self._pinpoint_coordinates(meas)
                self._find_slopes(lat, lon, self.ukf.P, map_data)  # updates SN, SE, Rfit

                # Calculate measurement
                h_asl_meas = meas.pos.h_asl[i]
                jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
                h_agl_meas = meas.pinpoint.range[i] * jac

                # Get map height at current location
                h_map_meas = RegularGridInterpolator(
                    (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)((lat, lon))
                self.traj.pos.h_agl[i] = h_agl_meas

                # Calculate measurement noise
                self._calc_rc(h_agl_meas)
                self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]
                self.ukf.R = self.params.R[i]

                # Store measurement
                self.params.Z[i] = h_asl_meas - h_agl_meas - h_map_meas

                # Store previous state for calculating corrections
                prev_x = self.ukf.x.copy()

                # Predict step
                self.ukf.predict()

                # Update step with measurement
                self.ukf.update(z=np.array([self.params.Z[i]]))

                # Extract state correction
                self.params.dX[:, i] = prev_x - self.ukf.x

                # Update trajectory with new state
                self._update_trajectory_from_state(meas, i)

                # Store covariance for later error analysis
                self.params.P_est[:, :, i] = self.ukf.P

                # Store approximate Kalman gain for analysis
                self.params.K[:, i] = self.params.dX[:, i] / self.params.Z[i] if self.params.Z[i] != 0 else 0

            except ValueError as e:
                if "out of bounds" in str(e):
                    print(f"Interpolation error at index {i}: {e}")
                    # Continue with next iteration
                else:
                    raise

        return self

    def _initialize_params(self):
        """
        Initialize filter parameters including covariance matrices.
        """
        self.params = UKFParams(self)

        # Initial error covariance
        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)

        # Process noise covariance
        self.params.Q = np.diag([0, 0, 0, 1e-6, 1e-6, 3e-6, 0, 0, 0, 3.33e-11, 3.33e-11, 3.33e-11])

    def _get_initial_state_vector(self, meas):
        """
        Get the initial state vector from measurements.

        Args:
            meas: Measurement trajectory

        Returns:
            numpy.ndarray: Initial state vector
        """
        return np.array([
            meas.pos.lat[0],
            meas.pos.lon[0],
            meas.pos.h_asl[0],
            meas.vel.north[0],
            meas.vel.east[0],
            meas.vel.down[0],
            meas.acc.north[0],
            meas.acc.east[0],
            meas.acc.down[0],
            meas.euler.psi[0],
            meas.euler.theta[0],
            meas.euler.phi[0]
        ])

    def _update_trajectory_from_state(self, meas, i):
        """
        Update the trajectory with the current state estimate.

        Args:
            meas: Measurement trajectory
            i: Current time step
        """
        # Extract state components
        self.traj.pos.lat[i] = self.ukf.x[0]
        self.traj.pos.lon[i] = self.ukf.x[1]
        self.traj.pos.h_asl[i] = self.ukf.x[2]
        self.traj.vel.north[i] = self.ukf.x[3]
        self.traj.vel.east[i] = self.ukf.x[4]
        self.traj.vel.down[i] = self.ukf.x[5]
        self.traj.acc.north[i] = self.ukf.x[6]
        self.traj.acc.east[i] = self.ukf.x[7]
        self.traj.acc.down[i] = self.ukf.x[8]
        self.traj.euler.psi[i] = self.ukf.x[9]
        self.traj.euler.theta[i] = self.ukf.x[10]
        self.traj.euler.phi[i] = self.ukf.x[11]

        # Update derived quantities
        self.traj.pos.h_map[i] = self.traj.pos.h_asl[i] - self.traj.pos.h_agl[i]
        self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]
        self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]