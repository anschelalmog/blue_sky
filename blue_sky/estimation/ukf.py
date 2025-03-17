import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from .base import BaseKalmanFilter
from ..utils.math import cosd, sind

class UKFParams:
    """
    Parameters for Unscented Kalman Filter.

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
        alpha, beta, kappa: UKF tuning parameters
        lambda_: Scaling parameter
        gamma: Scaling factor for sigma points
        weights_mean: Weights for mean calculation
        weights_covariance: Weights for covariance calculation
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

        # UKF-specific parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self._compute_lambda()
        self.gamma = np.sqrt(self.state_size + self.lambda_)
        self.weights_mean = self._compute_weights_mean()
        self.weights_covariance = self._compute_weights_covariance()

    def _compute_lambda(self):
        """
        Compute lambda parameter for UKF.

        Returns:
            float: Lambda parameter
        """
        return self.alpha ** 2 * (self.state_size + self.kappa) - self.state_size

    def _compute_weights_mean(self):
        """
        Compute weights for mean calculation in UKF.

        Returns:
            numpy.ndarray: Weights for mean
        """
        weights = np.full(2 * self.state_size + 1, 1 / (2 * (self.state_size + self.lambda_)))
        weights[0] = self.lambda_ / (self.state_size + self.lambda_)
        return weights

    def _compute_weights_covariance(self):
        """
        Compute weights for covariance calculation in UKF.

        Returns:
            numpy.ndarray: Weights for covariance
        """
        weights = np.full(2 * self.state_size + 1, 1 / (2 * (self.state_size + self.lambda_)))
        weights[0] = self.lambda_ / (self.state_size + self.lambda_) + (1 - self.alpha ** 2 + self.beta)
        return weights


class UKF(BaseKalmanFilter):
    """
    Unscented Kalman Filter implementation.

    UKF uses sigma points to capture the mean and covariance of the state,
    propagating these through nonlinear transformations to better handle nonlinearity.

    Attributes:
        Inherits attributes from BaseKalmanFilter
        params (UKFParams): Filter parameters
        epsilon (float): Small value for numerical stability
        measurement_size (int): Size of measurement vector
    """
    def __init__(self, args):
        """
        Initialize the UKF filter.

        Args:
            args: Configuration arguments
        """
        super().__init__(args)
        self.params = None  # Will be initialized in _initialize_params
        self.epsilon = 1e-6
        self.measurement_size = 1

    def run(self, map_data, meas):
        """
        Run the Unscented Kalman Filter to estimate trajectory.

        This method implements the main UKF algorithm:
        1. Initialization
        2. Sigma point generation
        3. Prediction step
        4. Measurement update
        5. State estimation

        Args:
            map_data: Map containing terrain information
            meas: Measurements to use for estimation

        Returns:
            self: Updated instance with estimation results
        """
        self._initialize_traj(meas)
        self._initialize_params()

        desc = "Estimating Trajectory with UKF"
        for i in tqdm(range(1, self.run_points), desc=desc):
            self.curr_state = i

            try:
                # Predict step
                X_sigma = self._compute_sigma_points()
                X_sigma_pred = self._predict_sigma_points(X_sigma, meas)
                x_pred = self._predict_state(X_sigma_pred)
                P_pred = self._predict_covariance(X_sigma_pred)

                # Update step
                lat, lon = self._pinpoint_coordinates(meas)
                self._find_slopes(lat, lon, P_pred, map_data)  # updates SN, SE, Rfit

                h_asl_meas = meas.pos.h_asl[i]
                jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
                h_agl_meas = meas.pinpoint.range[i] * jac

                # Get map height at current location
                h_map_meas = RegularGridInterpolator(
                    (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)((lat, lon))
                self.traj.pos.h_agl[i] = h_agl_meas

                self._calc_rc(h_agl_meas)
                self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]  # calc noise

                # Transform sigma points through measurement model
                Z_sigma = self._unscented_transform(X_sigma_pred, map_data)
                Z_pred = self._compute_predicted_measurement(Z_sigma)

                # Update state and covariance
                self._update_state(X_sigma_pred, Z_sigma, Z_pred, h_asl_meas, h_agl_meas, h_map_meas, meas)
                self._update_covariance(X_sigma_pred, Z_sigma, Z_pred)
            except ValueError as e:
                if "out of bounds" in str(e):
                    print(f"Interpolation error at index {i}: {e}")
                    # Continue with next iteration - could use previous state as fallback
                    self.traj.pos.lat[i] = self.traj.pos.lat[i-1]
                    self.traj.pos.lon[i] = self.traj.pos.lon[i-1]
                    # Copy other state variables from previous step
                else:
                    raise

        return self

    def _initialize_params(self):
        """
        Initialize UKF parameters including covariance matrices.
        """
        self.params = UKFParams(self)

        # Initial error covariance
        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)

        # Process noise covariance (very small in this implementation)
        self.params.Q = np.power(np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1e-7)

    def _compute_sigma_points(self):
        """
        Compute the sigma points for the current state.

        Returns:
            numpy.ndarray: Sigma points matrix
        """
        sigma_points = np.zeros((self.state_size, 2 * self.state_size + 1))

        # Get current state and covariance
        x = self._get_current_state_estimate()
        P = self._get_current_covariance_estimate()

        # Set the mean as the first sigma point
        sigma_points[:, 0] = x

        # Calculate square root of scaled covariance matrix
        try:
            sqrt_matrix = np.linalg.cholesky((self.state_size + self.params.lambda_) * P
                                             + self.epsilon * np.eye(self.state_size))

            # Generate remaining sigma points
            for i in range(self.state_size):
                sigma_points[:, i + 1] = x + sqrt_matrix[:, i]
                sigma_points[:, i + 1 + self.state_size] = x - sqrt_matrix[:, i]
        except np.linalg.LinAlgError:
            # If Cholesky decomposition fails, use a simple but less accurate approach
            print(f"Warning: Cholesky decomposition failed at step {self.curr_state}. Using diagonal approximation.")
            sqrt_vals = np.sqrt(np.diag(P) * (self.state_size + self.params.lambda_))
            for i in range(self.state_size):
                sigma_points[:, i + 1] = x.copy()
                sigma_points[:, i + 1][i] += sqrt_vals[i]
                sigma_points[:, i + 1 + self.state_size] = x.copy()
                sigma_points[:, i + 1 + self.state_size][i] -= sqrt_vals[i]

        return sigma_points

    def _predict_sigma_points(self, X_sigma, meas):
        """
        Propagate sigma points through the process model.

        Args:
            X_sigma: Sigma points matrix
            meas: Measurement trajectory

        Returns:
            numpy.ndarray: Predicted sigma points
        """
        i = self.curr_state
        X_sigma_pred = np.zeros_like(X_sigma)

        for j in range(2 * self.state_size + 1):
            # Position predictions with motion model
            X_sigma_pred[0, j] = X_sigma[0, j] + (
                    X_sigma[3, j] * self.del_t + 0.5 * meas.acc.north[i - 1] * self.del_t ** 2) / meas.mpd_north[i]
            X_sigma_pred[1, j] = X_sigma[1, j] + (
                    X_sigma[4, j] * self.del_t + 0.5 * meas.acc.east[i - 1] * self.del_t ** 2) / meas.mpd_east[i]
            X_sigma_pred[2, j] = X_sigma[2, j] + (
                    X_sigma[5, j] * self.del_t + 0.5 * meas.acc.down[i - 1] * self.del_t ** 2)

            # Velocity predictions with acceleration
            X_sigma_pred[3, j] = X_sigma[3, j] + meas.acc.north[i - 1] * self.del_t
            X_sigma_pred[4, j] = X_sigma[4, j] + meas.acc.east[i - 1] * self.del_t
            X_sigma_pred[5, j] = X_sigma[5, j] + meas.acc.down[i - 1] * self.del_t

            # Acceleration remains the same
            X_sigma_pred[6:9, j] = X_sigma[6:9, j]

            # Euler angles with linear change
            X_sigma_pred[9, j] = X_sigma[9, j] + ((meas.euler.psi[i] - meas.euler.psi[i - 1]) * self.del_t)
            X_sigma_pred[10, j] = X_sigma[10, j] + ((meas.euler.theta[i] - meas.euler.theta[i - 1]) * self.del_t)
            X_sigma_pred[11, j] = X_sigma[11, j] + ((meas.euler.phi[i] - meas.euler.phi[i - 1]) * self.del_t)

        return X_sigma_pred

    def _predict_state(self, X_sigma_pred):
        """
        Predict the state estimate based on sigma points.

        Args:
            X_sigma_pred: Predicted sigma points

        Returns:
            numpy.ndarray: Predicted state estimate
        """
        x_pred = np.zeros(self.state_size)
        for i in range(2 * self.state_size + 1):
            x_pred += self.params.weights_mean[i] * X_sigma_pred[:, i]
        return x_pred

    def _predict_covariance(self, X_sigma_pred):
        """
        Predict the error covariance based on sigma points.

        Args:
            X_sigma_pred: Predicted sigma points

        Returns:
            numpy.ndarray: Predicted error covariance
        """
        x_pred = self._predict_state(X_sigma_pred)
        P_pred = np.zeros((self.state_size, self.state_size))

        for i in range(2 * self.state_size + 1):
            diff = X_sigma_pred[:, i] - x_pred
            P_pred += self.params.weights_covariance[i] * np.outer(diff, diff)

        P_pred += self.params.Q
        return P_pred

    def _unscented_transform(self, X_sigma_pred, map_data):
        """
        Transform sigma points through the measurement model.

        Args:
            X_sigma_pred: Predicted sigma points
            map_data: Map data for terrain information

        Returns:
            numpy.ndarray: Transformed sigma points for measurements
        """
        i = self.curr_state
        Z_sigma = np.zeros((1, 2 * self.state_size + 1))

        try:
            for j in range(2 * self.state_size + 1):
                # Calculate measurement for each sigma point
                jac = cosd(X_sigma_pred[10, j]) * cosd(X_sigma_pred[11, j])
                h_agl_pred = self.traj.pinpoint.range[i] * jac

                # Get map height at predicted position
                lat_pred, lon_pred = X_sigma_pred[0, j], X_sigma_pred[1, j]
                h_map_pred = RegularGridInterpolator(
                    (map_data.axis['lat'], map_data.axis['lon']), map_data.grid)((lat_pred, lon_pred))

                # Predicted measurement
                Z_sigma[0, j] = X_sigma_pred[2, j] - h_agl_pred - h_map_pred
        except ValueError as e:
            if "out of bounds" in str(e):
                # Use last valid measurements for sigma points that cause errors
                print(f"Interpolation error in unscented transform at step {i}: {e}")
                Z_sigma.fill(0)  # Fill with zeros as fallback
            else:
                raise

        return Z_sigma

    def _compute_predicted_measurement(self, Z_sigma):
        """
        Compute the predicted measurement from transformed sigma points.

        Args:
            Z_sigma: Transformed sigma points for measurements

        Returns:
            numpy.ndarray: Predicted measurement
        """
        z_pred = np.zeros(self.measurement_size)
        for i in range(2 * self.state_size + 1):
            z_pred += self.params.weights_mean[i] * Z_sigma[:, i]
        return z_pred

    def _compute_cross_covariance(self, X_sigma_pred, Z_sigma, Z_pred):
        """
        Compute the cross-covariance between state and measurement.

        Args:
            X_sigma_pred: Predicted sigma points for state
            Z_sigma: Transformed sigma points for measurements
            Z_pred: Predicted measurement

        Returns:
            numpy.ndarray: Cross-covariance matrix
        """
        x_pred = self._predict_state(X_sigma_pred)
        Pxz = np.zeros((self.state_size, self.measurement_size))

        for i in range(2 * self.state_size + 1):
            Pxz += self.params.weights_covariance[i] * np.outer(
                X_sigma_pred[:, i] - x_pred, Z_sigma[:, i] - Z_pred)

        return Pxz

    def _compute_innovation_covariance(self, Z_sigma, Z_pred):
        """
        Compute the innovation (measurement) covariance.

        Args:
            Z_sigma: Transformed sigma points for measurements
            Z_pred: Predicted measurement

        Returns:
            numpy.ndarray: Innovation covariance matrix
        """
        Pzz = np.zeros((self.measurement_size, self.measurement_size))

        for i in range(2 * self.state_size + 1):
            Pzz += self.params.weights_covariance[i] * np.outer(
                Z_sigma[:, i] - Z_pred, Z_sigma[:, i] - Z_pred)

        # Add measurement noise
        Pzz += self.params.R[self.curr_state]
        return Pzz

    def _update_state(self, X_sigma_pred, Z_sigma, Z_pred, h_asl_meas, h_agl_meas, h_map_meas, meas):
        """
        Update the state estimate based on measurements.

        Args:
            X_sigma_pred: Predicted sigma points for state
            Z_sigma: Transformed sigma points for measurements
            Z_pred: Predicted measurement
            h_asl_meas: Measured height above sea level
            h_agl_meas: Measured height above ground level
            h_map_meas: Map height at measured position
            meas: Measurement trajectory
        """
        i = self.curr_state

        # Calculate cross-covariance and innovation covariance
        Pxz = self._compute_cross_covariance(X_sigma_pred, Z_sigma, Z_pred)
        Pzz = self._compute_innovation_covariance(Z_sigma, Z_pred)

        # Calculate Kalman gain
        try:
            K = Pxz @ np.linalg.pinv(Pzz)

            # Calculate innovation
            innovation = h_asl_meas - h_agl_meas - h_map_meas - Z_pred

            # State update
            self.params.dX[:, i] = K @ innovation

            # Apply state correction
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
        except np.linalg.LinAlgError:
            print(f"Warning: Matrix inversion failed in state update at step {i}. Using previous state.")
            # Fall back to previous state
            if i > 0:
                for attr in ['pos', 'vel', 'acc', 'euler']:
                    obj = getattr(self.traj, attr)
                    for key in vars(obj).keys():
                        if key != 'pinpoint' and hasattr(getattr(obj, key), '__getitem__'):
                            getattr(obj, key)[i] = getattr(obj, key)[i-1]

    def _update_covariance(self, X_sigma_pred, Z_sigma, Z_pred):
        """
        Update the error covariance based on the Kalman gain.

        Args:
            X_sigma_pred: Predicted sigma points for state
            Z_sigma: Transformed sigma points for measurements
            Z_pred: Predicted measurement
        """
        i = self.curr_state

        try:
            # Calculate cross-covariance and innovation covariance
            Pxz = self._compute_cross_covariance(X_sigma_pred, Z_sigma, Z_pred)
            Pzz = self._compute_innovation_covariance(Z_sigma, Z_pred)

            # Calculate Kalman gain
            K = Pxz @ np.linalg.pinv(Pzz)

            # Joseph form of covariance update for numerical stability
            P_pred = self._predict_covariance(X_sigma_pred)
            self.params.P_est[:, :, i] = P_pred - K @ Pzz @ K.T

            # Ensure positive definiteness
            self.params.P_est[:, :, i] = 0.5 * (self.params.P_est[:, :, i] + self.params.P_est[:, :, i].T)
            min_eig = np.min(np.real(np.linalg.eigvals(self.params.P_est[:, :, i])))

            if min_eig < 0:
                self.params.P_est[:, :, i] += (-min_eig + self.epsilon) * np.eye(self.state_size)
        except np.linalg.LinAlgError:
            print(f"Warning: Matrix inversion failed in covariance update at step {i}. Using predicted covariance.")
            if i > 0:
                self.params.P_est[:, :, i] = self.params.P_est[:, :, i-1]

    def _get_current_state_estimate(self):
        """
        Get the current state estimate as a vector.

        Returns:
            numpy.ndarray: Current state estimate
        """
        i = self.curr_state
        return np.array([
            self.traj.pos.lat[i],
            self.traj.pos.lon[i],
            self.traj.pos.h_asl[i],
            self.traj.vel.north[i],
            self.traj.vel.east[i],
            self.traj.vel.down[i],
            self.traj.acc.north[i],
            self.traj.acc.east[i],
            self.traj.acc.down[i],
            self.traj.euler.psi[i],
            self.traj.euler.theta[i],
            self.traj.euler.phi[i]
        ])

    def _get_current_covariance_estimate(self):
        """
        Get the current error covariance matrix.

        Returns:
            numpy.ndarray: Current error covariance matrix
        """
        return self.params.P_est[:, :, self.curr_state - 1] if self.curr_state > 0 else self.params.P_est[:, :, 0]