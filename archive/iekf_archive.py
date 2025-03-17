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
            self._predict_state(meas)  # Formula (2.1)
            p_pre = self._predict_covariance()  # Formula (2.2)

            # Inner iteration loop for refinement
            for iter_count in range(self.max_iter):
                # Measurement update
                lat, lon = self._pinpoint_coordinates(meas)

                try:
                    self._find_slopes(lat, lon, p_pre, map_data)  # updates SN, SE, Rfit

                    h_asl_meas = self.traj.pos.h_asl[i]  # meas.pos.h_asl[i]
                    jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
                    h_agl_meas = meas.pinpoint.range[i] * jac

                    h_map_meas_interp = RegularGridInterpolator(
                        (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
                    h_map_meas = h_map_meas_interp((lat, lon))
                    self.traj.pos.h_agl[i] = h_agl_meas

                    self.params.H[:3, i] = [-self.params.SN[i], -self.params.SE[i], -1]  # Update observation matrix
                    self._calc_rc(h_agl_meas)
                    self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]  # calc noise

                    # Estimation step
                    self._compute_gain(p_pre)
                    # Measurement Model
                    self.params.Z[i] = h_asl_meas - h_agl_meas - h_map_meas
                    self._estimate_covariance(p_pre)

                    self.params.dX[:, i] = self.params.K[:, i] * self.params.Z[i]
                    self._update_estimate_state(meas)

                    if self._check_convergence():
                        break
                except ValueError as e:
                    if "out of bounds" in str(e):
                        print(f"Interpolation error at index {i}: {e}")
                        # Continue with next iteration
                    else:
                        raise

        return self

    def _initialize_params(self):
        """
        Initialize filter parameters including covariance matrices and dynamics.
        """
        self.params = IEKFParams(self)

        # Initial error in pos(north, east, down) in[m],
        # vel(north, east, down) in [m/s],
        # acc(north, east, down) in [m/s^2],
        # euler(yaw, pitch, roll) in [deg]
        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)

        # Process noise covariance
        self.params.Q = np.diag([0, 0, 0, 1e-6, 1e-6, 3e-6, 0, 0, 0, 3.33e-11, 3.33e-11, 3.33e-11])

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
        Compute estimated velocities and positions based on previous and current measurements.

        Formula: x_hat(t|t-1) = F * x_hat(t-1|t-1) + B * u(t)

        Args:
            meas: Measurement trajectory
        """
        i = self.curr_state

        # Update positions based on velocity and acceleration
        self.traj.pos.lat[i] = (self.traj.pos.lat[i - 1] +
                                (self.traj.vel.north[i - 1] * self.del_t +
                                 0.5 * self.traj.acc.north[i - 1] * self.del_t ** 2) / meas.mpd_north[i])
        self.traj.pos.lon[i] = (self.traj.pos.lon[i - 1] +
                                (self.traj.vel.east[i - 1] * self.del_t +
                                 0.5 * self.traj.acc.east[i - 1] * self.del_t ** 2) / meas.mpd_east[i])
        self.traj.pos.h_asl[i] = (self.traj.pos.h_asl[i - 1] +
                                  (self.traj.vel.down[i - 1] * self.del_t +
                                   0.5 * self.traj.acc.down[i - 1] * self.del_t ** 2))

        # Update velocities based on acceleration
        self.traj.vel.north[i] = self.traj.vel.north[i - 1] + meas.acc.north[i - 1] * self.del_t
        self.traj.vel.east[i] = self.traj.vel.east[i - 1] + meas.acc.east[i - 1] * self.del_t
        self.traj.vel.down[i] = self.traj.vel.down[i - 1] + meas.acc.down[i - 1] * self.del_t

        # Assuming linear change in Euler angles
        self.traj.euler.psi[i] = self.traj.euler.psi[i - 1] + ((meas.euler.psi[i] - meas.euler.psi[i - 1]) * self.del_t)
        self.traj.euler.theta[i] = self.traj.euler.theta[i - 1] + (
                (meas.euler.theta[i] - meas.euler.theta[i - 1]) * self.del_t)
        self.traj.euler.phi[i] = self.traj.euler.phi[i - 1] + ((meas.euler.phi[i] - meas.euler.phi[i - 1]) * self.del_t)

        # Direct velocity update from measurements
        self.traj.vel.north[i] = self.traj.vel.north[i - 1] + meas.vel.north[i] - meas.vel.north[i - 1]
        self.traj.vel.east[i] = self.traj.vel.east[i - 1] + meas.vel.east[i] - meas.vel.east[i - 1]
        self.traj.vel.down[i] = self.traj.vel.down[i - 1] + meas.vel.down[i] - meas.vel.down[i - 1]

        # Direct position update based on velocity
        self.traj.pos.lat[i] = self.traj.pos.lat[i - 1] + self.traj.vel.north[i] / meas.mpd_north[i] * self.del_t
        self.traj.pos.lon[i] = self.traj.pos.lon[i - 1] + self.traj.vel.east[i] / meas.mpd_east[i] * self.del_t
        self.traj.pos.h_asl[i] = self.traj.pos.h_asl[i - 1] + meas.pos.h_asl[i] - meas.pos.h_asl[i - 1]

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
        H = self.params.H[:, i]
        R = self.params.R[i]

        self.params.K[:, i] = P @ H.T / (H @ P @ H.T + R)

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
            meas: Measurement trajectory
        """
        i = self.curr_state

        # Update state with correction
        self.traj.pos.lat[i] -= self.params.dX[0, i] / meas.mpd_north[i]
        self.traj.pos.lon[i] -= self.params.dX[1, i] / meas.mpd_east[i]
        self.traj.pos.h_asl[i] -= self.params.dX[2, i]
        self.traj.vel.north[i] -= self.params.dX[3, i]
        self.traj.vel.east[i] -= self.params.dX[4, i]
        self.traj.vel.down[i] -= self.params.dX[5, i]
        self.traj.acc.north[i] -= self.params.dX[6, i]
        self.traj.acc.east[i] -= self.params.dX[7, i]
        self.traj.acc.down[i] -= self.params.dX[8, i]
        self.traj.euler.psi[i] -= self.params.dX[9, i]
        self.traj.euler.theta[i] -= self.params.dX[10, i]
        self.traj.euler.phi[i] -= self.params.dX[11, i]

        # Update derived quantities
        self.traj.pos.h_map[i] = self.traj.pos.h_asl[i] - self.traj.pos.h_agl[i]
        self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]
        self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]

    def _check_convergence(self):
        """
        Check if the IEKF iteration has converged based on state change.

        Returns:
            bool: True if converged, False otherwise
        """
        i = self.curr_state
        if i <= 1:
            return False

        loss = np.linalg.norm(self.params.dX[:, i] - self.params.dX[:, i - 1])
        return loss < self.conv_rate