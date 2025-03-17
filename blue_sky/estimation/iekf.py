import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time

from .base import BaseKalmanFilter
from ..utils.math import cosd
from ..utils.progress import progress_bar


class IEKFParams:
    """
    Parameters for Iterative Extended Kalman Filter.

    This class stores all the matrices and parameters needed for IEKF estimation.

    Attributes:
        P_est: Error covariance matrix
        Q: Process noise covariance
        Phi: State transition matrix
        Phi_dot: Derivative of state transition matrix
        H: Observation matrix
        Z: Measurement vector
        R: Measurement noise covariance
        Rc: Range-dependent component of R
        Rfit: Fit error component of R
        K: Kalman gain
        dX: State correction
        SN: North slope
        SE: East slope
    """
    def __init__(self, kf):
        """
        Initialize IEKF parameters.

        Args:
            kf: Kalman filter instance containing configuration
        """
        self.P_est = np.zeros((kf.state_size, kf.state_size, kf.run_points))
        self.Q = np.zeros((kf.state_size, kf.state_size))  # system noise, constant
        self.Phi = np.zeros((kf.state_size, kf.state_size))
        self.Phi_dot = np.zeros((kf.state_size, kf.state_size))
        self.H = np.zeros((kf.state_size, kf.run_points))  # observation vector
        self.Z = np.zeros(kf.run_points)  # measurement vector
        self.R = np.zeros(kf.run_points)
        self.Rc = np.zeros(kf.run_points)
        self.Rfit = np.zeros(kf.run_points)
        self.K = np.zeros((kf.state_size, kf.run_points))
        self.dX = np.zeros((kf.state_size, kf.run_points))
        self.SN = np.zeros(kf.run_points)
        self.SE = np.zeros(kf.run_points)


class IEKF(BaseKalmanFilter):
    """
    Iterative Extended Kalman Filter implementation.

    IEKF iteratively refines the state estimate through multiple update iterations
    within each prediction-update cycle, improving accuracy for nonlinear systems.

    Attributes:
        Inherits attributes from BaseKalmanFilter
        max_iter (int): Maximum number of iterations per step
        conv_rate (float): Convergence rate threshold
        params (IEKFParams): Filter parameters
    """
    def __init__(self, args):
        """
        Initialize the IEKF filter.

        Args:
            args: Configuration arguments
        """
        super().__init__(args)
        self.max_iter = 1  # Maximum iterations per step
        self.conv_rate = 0  # Convergence rate threshold
        self.params = None  # Will be initialized in _initialize_params

    def run(self, map_data, meas):
        """
        Run the Iterative Extended Kalman Filter to estimate trajectory.

        This method implements the main loop for IEKF, including:
        1. Initialization
        2. Prediction step
        3. Iterative measurement update
        4. State correction

        Args:
            map_data: Map containing terrain information
            meas: Measurements to use for estimation

        Returns:
            self: Updated instance with estimation results
        """
        self._initialize_traj(meas)
        self._initialize_params()

        time.sleep(0.01)  # Small delay for progress bar
        desc = "Estimating Trajectory with IEKF"

        for i in progress_bar(range(1, self.run_points), desc):
            self.curr_state = i

            # Prediction step
            self._predict_state(meas)
            p_pre = self._predict_covariance()

            # Store initial state estimate for iteration
            initial_state = {
                'lat': self.traj.pos.lat[i],
                'lon': self.traj.pos.lon[i],
                'h_asl': self.traj.pos.h_asl[i]
            }

            # Clear the previous iteration's state
            if hasattr(self, 'prev_dX'):
                delattr(self, 'prev_dX')

            # Inner iteration loop for refinement
            converged = False

            for iter_count in range(self.max_iter):
                try:
                    # Measurement update
                    lat, lon = self._pinpoint_coordinates(meas)

                    # Check if coordinates are within map boundaries
                    lat_min, lat_max = np.min(map_data.axis['lat']), np.max(map_data.axis['lat'])
                    lon_min, lon_max = np.min(map_data.axis['lon']), np.max(map_data.axis['lon'])

                    if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                        print(f"Warning: Point {i} outside map boundaries. Skipping update.")
                        break

                    # Find terrain slopes and fit error
                    self._find_slopes(lat, lon, p_pre, map_data)

                    # Get height measurements
                    h_asl_meas = meas.pos.h_asl[i]
                    jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
                    h_agl_meas = meas.pinpoint.range[i] * jac

                    # Get terrain height at pinpoint location
                    interpolator = RegularGridInterpolator(
                        (map_data.axis['lat'], map_data.axis['lon']), map_data.grid,
                        bounds_error=False, fill_value=None)
                    h_map_meas = interpolator((lat, lon)).item()

                    # Store height above ground
                    self.traj.pos.h_agl[i] = h_agl_meas
                    self.traj.pinpoint.h_map[i] = h_map_meas

                    # Update measurement model
                    self.params.H[:3, i] = [-self.params.SN[i], -self.params.SE[i], -1]

                    # Calculate measurement noise
                    self._calc_rc(h_agl_meas)
                    self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]

                    # Compute innovation (measurement residual)
                    # h_asl = h_map + h_agl
                    h_pred = h_map_meas + h_agl_meas  # Predicted height ASL
                    self.params.Z[i] = h_asl_meas - h_pred  # Measurement residual

                    # Compute gain and update state estimate
                    self._compute_gain(p_pre)
                    self._estimate_covariance(p_pre)
                    self.params.dX[:, i] = self.params.K[:, i] * self.params.Z[i]
                    self._update_estimate_state(meas)

                    # Check for convergence
                    converged = self._check_convergence()
                    if converged:
                        print(f"Converged after {iter_count+1} iterations at step {i}")
                        break

                except Exception as e:
                    print(f"Error in IEKF iteration {iter_count} at step {i}: {e}")
                    # Restore initial state estimate if iterations fail
                    self.traj.pos.lat[i] = initial_state['lat']
                    self.traj.pos.lon[i] = initial_state['lon']
                    self.traj.pos.h_asl[i] = initial_state['h_asl']
                    break

            # If no convergence after max iterations, give a warning
            if not converged and self.max_iter > 1:
                print(f"Warning: No convergence after {self.max_iter} iterations at step {i}")

        return self

    def _initialize_params(self):
        """
        Initialize filter parameters including covariance matrices and dynamics.
        """
        self.params = IEKFParams(self)
        self.max_iter = 5  # Allow multiple iterations for IEKF
        self.conv_rate = 0.1  # Set meaningful convergence threshold

        # Initial error in pos(north, east, down) in[m],
        # vel(north, east, down) in [m/s],
        # acc(north, east, down) in [m/s^2],
        # euler(yaw, pitch, roll) in [deg]
        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)

        # Process noise covariance
        self.params.Q = np.diag([0.1, 0.1, 0.1, 1e-6, 1e-6, 3e-6, 1e-8, 1e-8, 1e-8, 3.33e-11, 3.33e-11, 3.33e-11])

        # Dynamic Equation: dX_k+1 = Phi_k+1|k * dX_k + W_k+1
        self.params.Phi = np.eye(self.state_size)
        self.params.Phi[0, 3] = self.params.Phi[1, 4] = self.params.Phi[2, 5] = self.del_t
        self.params.Phi[0, 6] = self.params.Phi[1, 7] = self.params.Phi[2, 8] = 0.5 * self.del_t ** 2
        self.params.Phi[3, 6] = self.params.Phi[4, 7] = self.params.Phi[5, 8] = self.del_t
        self.params.Phi[9, 10] = self.params.Phi[10, 11] = self.del_t

        # dPhi/dt
        self.params.Phi_dot[0, 3] = self.params.Phi_dot[1, 4] = self.params.Phi_dot[2, 5] = 1
        self.params.Phi_dot[0, 6] = self.params.Phi_dot[1, 7] = self.params.Phi_dot[2, 8] = self.del_t
        self.params.Phi_dot[3, 6] = self.params.Phi_dot[4, 7] = self.params.Phi_dot[5, 8] = 1
        self.params.Phi_dot[9, 10] = self.params.Phi_dot[10, 11] = 1

    def _predict_state(self, meas):
        """
        Compute estimated velocities and positions based on previous state and system dynamics.

        This implements the prediction step of the Kalman filter using kinematic equations.

        Args:
            meas: Measurement trajectory (used for conversion factors)
        """
        i = self.curr_state

        # Use proper kinematic equations for state prediction
        # Position update based on previous velocity and acceleration
        self.traj.pos.lat[i] = (self.traj.pos.lat[i - 1] +
                                (self.traj.vel.north[i - 1] * self.del_t +
                                 0.5 * self.traj.acc.north[i - 1] * self.del_t**2) / meas.mpd_north[i])

        self.traj.pos.lon[i] = (self.traj.pos.lon[i - 1] +
                                (self.traj.vel.east[i - 1] * self.del_t +
                                 0.5 * self.traj.acc.east[i - 1] * self.del_t**2) / meas.mpd_east[i])

        self.traj.pos.h_asl[i] = (self.traj.pos.h_asl[i - 1] -
                                  (self.traj.vel.down[i - 1] * self.del_t +
                                   0.5 * self.traj.acc.down[i - 1] * self.del_t**2))

        # Velocity update based on acceleration
        self.traj.vel.north[i] = self.traj.vel.north[i - 1] + self.traj.acc.north[i - 1] * self.del_t
        self.traj.vel.east[i] = self.traj.vel.east[i - 1] + self.traj.acc.east[i - 1] * self.del_t
        self.traj.vel.down[i] = self.traj.vel.down[i - 1] + self.traj.acc.down[i - 1] * self.del_t

        # Use linear interpolation for Euler angles based on rates
        self.traj.euler.psi[i] = self.traj.euler.psi[i - 1] + ((meas.euler.psi[i] - meas.euler.psi[i - 1]) / self.del_t) * self.del_t
        self.traj.euler.theta[i] = self.traj.euler.theta[i - 1] + ((meas.euler.theta[i] - meas.euler.theta[i - 1]) / self.del_t) * self.del_t
        self.traj.euler.phi[i] = self.traj.euler.phi[i - 1] + ((meas.euler.phi[i] - meas.euler.phi[i - 1]) / self.del_t) * self.del_t

        # Update derived quantities
        self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]
        self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]

    def _predict_covariance(self):
        """
        Compute prior error covariance matrix.

        Formula: P(t|t-1) = F * P(t-1|t-1) * F^T + Q

        Returns:
            numpy.ndarray: Prior error covariance matrix
        """
        return self.params.Phi @ self.params.P_est[:, :, self.curr_state - 1] @ self.params.Phi.T + self.params.Q

    def _compute_gain(self, P):
        """
        Compute Kalman gain matrix.

        Formula: K = P * H^T * (H * P * H^T + R)^-1

        Args:
            P: Prior error covariance matrix
        """
        i = self.curr_state
        H = self.params.H[:, i].reshape(1, -1)  # Ensure proper shape for matrix multiplication
        R = np.array([[self.params.R[i]]])  # Scalar to matrix

        # Calculate innovation covariance with numerical stability
        S = H @ P @ H.T + R

        # Ensure numerical stability in matrix inversion
        if np.linalg.det(S) < 1e-10:
            # Add small regularization term
            S += 1e-10 * np.eye(S.shape[0])

        # Calculate Kalman gain using matrix operations
        self.params.K[:, i] = (P @ H.T @ np.linalg.inv(S)).flatten()

    def _estimate_covariance(self, P):
        """
        Update error covariance matrix after state update.

        Formula: P(t|t) = (I - K * H) * P(t|t-1) * (I - K * H)^T + K * R * K^T

        Args:
            P: Prior error covariance matrix
        """
        i = self.curr_state
        K = self.params.K[:, i].reshape(-1, 1)
        H = np.transpose(self.params.H[:, i].reshape(-1, 1))
        R = np.array([[self.params.R[i]]])
        I_mat = np.eye(self.state_size)

        # Joseph Formula for numerical stability
        self.params.P_est[:, :, i] = (I_mat - K @ H) @ P @ (I_mat - K @ H).T + K @ R @ K.T

        # Regularization to prevent singularity
        epsilon = 1e-6  # Small positive number
        self.params.P_est[:, :, i] += epsilon * np.eye(self.state_size)

    def _update_estimate_state(self, meas):
        """
        Update state estimate based on Kalman gain and innovation.

        Args:
            meas: Measurement trajectory for reference data
        """
        i = self.curr_state

        # Update state with correction
        # Apply state corrections with proper scaling
        self.traj.pos.lat[i] += self.params.dX[0, i] / meas.mpd_north[i]
        self.traj.pos.lon[i] += self.params.dX[1, i] / meas.mpd_east[i]
        self.traj.pos.h_asl[i] += self.params.dX[2, i]
        self.traj.vel.north[i] += self.params.dX[3, i]
        self.traj.vel.east[i] += self.params.dX[4, i]
        self.traj.vel.down[i] += self.params.dX[5, i]
        self.traj.acc.north[i] += self.params.dX[6, i]
        self.traj.acc.east[i] += self.params.dX[7, i]
        self.traj.acc.down[i] += self.params.dX[8, i]
        self.traj.euler.psi[i] += self.params.dX[9, i]
        self.traj.euler.theta[i] += self.params.dX[10, i]
        self.traj.euler.phi[i] += self.params.dX[11, i]

        # Update derived quantities
        self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]
        self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]

        # Update height above ground level
        self.traj.pos.h_agl[i] = self.traj.pos.h_asl[i] - self.traj.pinpoint.h_map[i]

    def _check_convergence(self):
        """
        Check if the IEKF iteration has converged based on state change magnitude.

        Returns:
            bool: True if converged, False otherwise
        """
        i = self.curr_state

        # For first iteration, no convergence yet
        if not hasattr(self, 'prev_dX'):
            self.prev_dX = self.params.dX[:, i].copy()
            return False

        # Calculate relative change between current and previous iteration
        diff = np.linalg.norm(self.params.dX[:, i] - self.prev_dX)
        rel_change = diff / (np.linalg.norm(self.prev_dX) + 1e-10)

        # Store current update for next iteration
        self.prev_dX = self.params.dX[:, i].copy()

        # Check if relative change is below threshold
        return rel_change < self.conv_rate