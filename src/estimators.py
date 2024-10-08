import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import matplotlib.pyplot as plt
from icecream import ic
import time

from src.utils import cosd, sind, DCM, progress_bar
from src.base_traj import BaseTraj
from src.pinpoint_calc import PinPoint


def jac_north(psi, theta, phi):
    return cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi)


def jac_east(psi, theta, phi):
    return sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi)


class IEKF:
    def __init__(self, args):
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        self.del_t = args.time_res
        self.state_size = args.kf_state_size
        self.curr_state = None
        self.max_iter = 1  # if args.max_iter is None else args.max_iter
        self.conv_rate = 0  # if args.conv_rate is None else args.conv_rate
        self.traj = BaseTraj(self.run_points)
        self.traj.pinpoint = PinPoint(self.run_points)
        self.params = IEKFParams(self)

    def run(self, map_data, meas):
        """
        This method implements the main loop for the Iterative Extended Kalman Filter (IEKF).
        IEKF is used here for estimating the trajectory of a vehicle using sensor measurements and map data.

        KF Algorithm:
        1. Initialization: the state estimates and covariance matrices.
        2. Prediction Step:
           - State Prediction:
             Formula (2.1): x_hat_k|k-1 = F * x_hat_k-1|k-1 + B * u_k
             x_hat - state estimate, F - transition model, B is the control-input model,
             and u is the control vector.
           - Covariance Prediction:
             Formula (2.2): P_k|k-1 = F_k * P_k-1|k-1 * F_k^T + Q_k
             Where P is the covariance matrix of the state estimate, and Q is the process noise covariance.

        3. Update Step:
            - Measurement Update: Calculate the Kalman Gain and update the estimate with the measurement.
              Formula (3.1): K_k = P_k|k-1 * H_k^T / (H_k * P_k|k-1 * H_k^T + R_k)
              Where K is the Kalman Gain, H is the observation model, and R is the measurement noise covariance.
            - State Update:
              Formula (3.2): x_hat_k|k = x_hat_k|k-1 + K_k * (z_k - h(x_hat_k|k-1))
              Where z is the actual measurement, and h is the measurement function.
            - Covariance Update:
              Formula (3.3): P_k|k = (I - K_k * H_k) * P_k|k-1
              Where I is the identity matrix.

        This method iterates over the measurement points, applying these steps to estimate the vehicle's
        trajectory.

        Parameters:
        - map_data: Contains the map information necessary for navigation.
        - meas: Sensor measurements used for estimating the trajectory.

        Returns:
        - self: An instance of the IEKF class with updated trajectory estimates.
        """

        self._initialize_traj(meas)
        self._initialize_params()

        time.sleep(0.01)
        desc = "Estimating Trajectory with IEKF"
        # for i in
        for i in progress_bar(self.run_points, desc):
            self.curr_state = i

            "-Prediction step"
            self._predict_state(meas)  # Formula (2.1)
            p_pre = self._predict_covariance()  # Formula (2.2)

            # inner_loop = 'estimation iterations'
            # for iter in tqdm(range(self.max_iter), desc = inner_loop):
            for iter in range(self.max_iter):
                "-Measurement update"
                # H_x = self._compute_measurement
                lat, lon = self._pinpoint_coordinates(meas)
                self._find_slopes(lat, lon, p_pre, map_data)  # updates SN, SE, Rfit #
                h_asl_meas = self.traj.pos.h_asl[i]  # meas.pos.h_asl[i]
                jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
                h_agl_meas = meas.pinpoint.range[i] * jac

                h_map_meas_interp = RegularGridInterpolator((map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
                h_map_meas = h_map_meas_interp((lat, lon))
                self.traj.pos.h_agl[i] = h_agl_meas

                self.params.H[:3, i] = [-self.params.SN[i], -self.params.SE[i], -1]  # Update observation matrix
                self._calc_rc(h_agl_meas)
                self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]  # calc noise

                "-Estimation step"
                self._compute_gain(p_pre)
                # Measurement Model
                self.params.Z[i] = h_asl_meas - h_agl_meas - h_map_meas
                # self.params.Z[i] = self.traj.pos.h_asl[i] - h_agl_meas - h_map_meas
                self._estimate_covariance(p_pre)

                self.params.dX[:, i] = self.params.K[:, i] * self.params.Z[i]
                self._update_estimate_state(meas)

                if self.check_convergence():
                    break

        return self

    def _initialize_traj(self, meas):
        """
        Set initial values from measurements

        """
        self.traj.pos.east[0] = meas.pos.east[0]
        self.traj.pos.north[0] = meas.pos.north[0]
        self.traj.pos.lat[0] = meas.pos.lat[0]
        self.traj.pos.lon[0] = meas.pos.lon[0]
        self.traj.pos.h_asl[0] = meas.pos.h_asl[0]  # altitude
        jac = cosd(meas.euler.theta[0]) * cosd(meas.euler.phi[0])
        self.traj.pos.h_agl[0] = meas.pinpoint.range[0] * jac  # H_agl_p
        #
        self.traj.vel.north[0] = meas.vel.north[0]
        self.traj.vel.east[0] = meas.vel.east[0]
        self.traj.vel.down[0] = meas.vel.down[0]
        #
        self.traj.pinpoint.delta_north[0] = meas.pinpoint.delta_north[0]
        self.traj.pinpoint.delta_east[0] = meas.pinpoint.delta_east[0]
        self.traj.pinpoint.h_map[0] = meas.pinpoint.h_map[0]
        #
        self.traj.euler.psi[0] = meas.euler.psi[0]
        self.traj.euler.theta[0] = meas.euler.theta[0]
        self.traj.euler.phi[0] = meas.euler.phi[0]

    def _initialize_params(self):
        self.params = IEKFParams(self)
        # initial error in pos(north, east, down) in[m],
        #                  vel(north, east, down) in [m/s],
        #                  acc(north, east, down) in [m/s^2],
        #                  euler(yaw, pitch,roll) in [deg]
        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)
        #
        self.params.Q = np.diag([0, 0, 0, 1e-6, 1e-6, 3e-6, 0, 0, 0, 3.33e-11, 3.33e-11, 3.33e-11])

        # Dynamic Equation:
        # dX_k + 1 = Phi_k + 1 | k * dX_k + W_k + 1
        self.params.Phi = np.eye(self.state_size)
        self.params.Phi[0, 3] = self.params.Phi[1, 4] = self.params.Phi[2, 5] = self.del_t
        self.params.Phi[0, 6] = self.params.Phi[1, 7] = self.params.Phi[2, 8] = 0.5 * self.del_t ** 2
        self.params.Phi[3, 6] = self.params.Phi[4, 7] = self.params.Phi[5, 8] = self.del_t
        self.params.Phi[9, 10] = self.params.Phi[10, 11] = self.del_t

        # dPhi/ dt
        self.params.Phi_dot[0, 3] = self.params.Phi_dot[1, 4] = self.params.Phi_dot[2, 5] = 1
        self.params.Phi_dot[0, 6] = self.params.Phi_dot[1, 7] = self.params.Phi_dot[2, 8] = self.del_t
        self.params.Phi_dot[3, 6] = self.params.Phi_dot[4, 7] = self.params.Phi_dot[5, 8] = 1
        self.params.Phi_dot[9, 10] = self.params.Phi_dot[10, 11] = 1

    def _predict_state(self, meas):
        """
        compute estimated velocities and positions based on previous and current measurements

        x_hat(t|t-1) = F * x_hat(t-1|t-1) + B * u(t)
        :param meas: measurement trajectory
        :return: updating class
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
        # self.traj.vel.north[i] = self.traj.vel.north[i - 1] + self.traj.acc.north[i - 1] * self.del_t
        self.traj.vel.north[i] = self.traj.vel.north[i - 1] + meas.acc.north[i - 1] * self.del_t
        # self.traj.vel.east[i] = self.traj.vel.east[i - 1] + self.traj.acc.east[i - 1] * self.del_t
        self.traj.vel.east[i] = self.traj.vel.east[i - 1] + meas.acc.east[i - 1] * self.del_t
        # self.traj.vel.down[i] = self.traj.vel.down[i - 1] + self.traj.acc.down[i - 1] * self.del_t
        self.traj.vel.down[i] = self.traj.vel.down[i - 1] + meas.acc.down[i - 1] * self.del_t

        # Assuming linear change in Euler angles
        self.traj.euler.psi[i] = self.traj.euler.psi[i - 1] + ((meas.euler.psi[i] - meas.euler.psi[i - 1]) * self.del_t)
        self.traj.euler.theta[i] = self.traj.euler.theta[i - 1] + (
                (meas.euler.theta[i] - meas.euler.theta[i - 1]) * self.del_t)
        self.traj.euler.phi[i] = self.traj.euler.phi[i - 1] + ((meas.euler.phi[i] - meas.euler.phi[i - 1]) * self.del_t)

        self.traj.vel.north[i] = self.traj.vel.north[i - 1] + meas.vel.north[i] - meas.vel.north[i - 1]
        self.traj.vel.east[i] = self.traj.vel.east[i - 1] + meas.vel.east[i] - meas.vel.east[i - 1]
        self.traj.vel.down[i] = self.traj.vel.down[i - 1] + meas.vel.down[i] - meas.vel.down[i - 1]

        self.traj.pos.lat[i] = self.traj.pos.lat[i - 1] + self.traj.vel.north[i] / meas.mpd_north[i] * self.del_t
        self.traj.pos.lon[i] = self.traj.pos.lon[i - 1] + self.traj.vel.east[i] / meas.mpd_east[i] * self.del_t
        self.traj.pos.h_asl[i] = self.traj.pos.h_asl[i - 1] + meas.pos.h_asl[i] - meas.pos.h_asl[i - 1]

    def _predict_covariance(self):
        """
        Compute prior Error covariance matrix

        :return: P(t|t-1) = F * P(t-1|t-1) * F^T + Q
        """
        return self.params.Phi @ self.params.P_est[:, :, self.curr_state - 1] @ self.params.Phi.T + self.params.Q

    def _pinpoint_coordinates(self, meas):
        """
        calculate current lat, lon

        :param meas:
        :return:
        """
        i = self.curr_state
        psi = meas.euler.psi[i]
        theta = meas.euler.theta[i]
        phi = meas.euler.phi[i]

        self.traj.pinpoint.delta_north[i] = meas.pinpoint.range[i] * (
                cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi))
        self.traj.pinpoint.delta_east[i] = meas.pinpoint.range[i] * (
                sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi))

        # lon - lat at pinpoint
        lat = self.traj.pos.lat[i] + self.traj.pinpoint.delta_north[i] / meas.mpd_north[i]
        lon = self.traj.pos.lon[i] + self.traj.pinpoint.delta_east[i] / meas.mpd_east[i]

        lat, lon = meas.pos.lat[i], meas.pos.lon[i]

        return lat, lon

    def _find_slopes(self, lat, lon, p_pre, map_data):
        """
        calculate the north-south and east-west slopes of terrain at a given lat lon,
        and estimate the Error of the slope model

        :param lat: Latitude at point [deg]
        :param lon: Longitude at point [deg]
        :param p_pre: A priori Error covariance matrix
        :param map_data: MapLoad instance that contain the grid and it's axes
        :return: SN, SE: slopes values , RFIT: fit Error
        """
        i = self.curr_state
        dP = 100  # distance increments in [m]
        delPmap = np.array([dP / map_data.mpd['north'][i], dP / map_data.mpd['east'][i]])  # [deg]

        # max number of points in each direction
        maxP = np.sqrt(max(p_pre[0][0], p_pre[1][1]))
        KP = 3
        NC = np.ceil(max(KP, 2 * np.ceil(KP * maxP / dP) + 1) / 2)
        idx = (NC - 1) / 2  # indices

        # create lat lon vectors according to grid indices
        pos_offset = np.arange(-idx, idx + 1)
        lat_vec, lon_vec = delPmap[0] * pos_offset + lat, delPmap[0] * pos_offset + lon
        xp, yp = np.meshgrid(lon_vec, lat_vec)
        # scaling factors for slope calc
        sx2 = (dP ** 2) * 2 * NC * np.sum(np.power(np.arange(1, idx + 1), 2))
        sy2 = sx2

        # interpolate elevation data for current location
        interpolator = RegularGridInterpolator((map_data.axis['lat'], map_data.axis['lon']), map_data.grid)
        ref_elevation = interpolator((lat, lon))
        grid_elevations = interpolator((yp, xp))

        # calculate slopes in x and y directions
        syh = dP * np.dot(pos_offset, grid_elevations - ref_elevation).sum()
        sxh = dP * np.dot(grid_elevations - ref_elevation, pos_offset).sum()

        self.params.SN[i], self.params.SE[i] = sxh / sx2, syh / sy2

        SN, SE = self.params.SN[i], self.params.SE[i]
        MP = (2 * idx + 1) ** 2  # number of points in the mesh grid
        In = 0
        # calculate the Error over the grid
        for step_E in np.arange(-idx, idx + 1):
            for step_N in np.arange(-idx, idx + 1):
                north = int(step_N + idx)
                east = int(step_E + idx)
                In = In + (dP * (SN * step_N + SE * step_N) - grid_elevations[east][north] + ref_elevation) ** 2
        self.params.Rfit[i] = In / (MP - 1)

    def _calc_rc(self, h_agl_meas):
        if h_agl_meas <= 200:
            self.params.Rc[self.curr_state] = 100
        elif h_agl_meas <= 760:
            self.params.Rc[self.curr_state] = 225  # 100 + 125
        elif h_agl_meas <= 1000:
            self.params.Rc[self.curr_state] = 400  # 225 + 175
        elif h_agl_meas <= 5000:
            self.params.Rc[self.curr_state] = 1000  # 400 + 600
        elif h_agl_meas <= 7000:
            self.params.Rc[self.curr_state] = 1500  # 1000 + 500
        else:
            self.params.Rc[self.curr_state] = 3000  # 1500 + 1500

        # Increase Rc towards the end to account for increasing measurement uncertainty
        if self.curr_state > self.run_points * 0.8:
            self.params.Rc[self.curr_state] *= 1.5

    def _compute_gain(self, P):
        i = self.curr_state
        H = self.params.H[:, i]
        R = self.params.R[i]

        self.params.K[:, i] = P @ H.T / (H @ P @ H.T + R)

    def _estimate_covariance(self, P):
        i = self.curr_state
        K = self.params.K[:, i].reshape(-1, 1)
        H = np.transpose(self.params.H[:, i].reshape(-1, 1))
        R = np.array([[self.params.R[i]]])
        I_mat = np.eye(self.state_size)

        # Joseph Formula
        self.params.P_est[:, :, i] = (I_mat - K @ H) @ P @ (I_mat - K @ H).T + K @ R @ K.T

        # Optimal
        #  self.params.P_est[:, :, i] = (I- K @ H) @ P

        # Regularization
        epsilon = 1e-6  # Small positive number
        self.params.P_est[:, :, i] += epsilon * np.eye(self.state_size)

    def _update_estimate_state(self, meas):
        # Update estimated state
        i = self.curr_state

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
        self.traj.pos.h_map[i] = self.traj.pos.h_asl[i] - self.traj.pos.h_agl[i]
        self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]
        self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]

    def check_convergence(self) -> bool:
        """
           Check if the IEKF state estimate has converged using the Euclidean norm.

           Parameters:
           - prev_state: The state estimate from the previous iteration (numpy array).
           - curr_state: The updated state estimate from the current iteration (numpy array).
           - threshold: The convergence threshold; the iteration is considered converged
                        if the Euclidean norm of the change in the state estimate is less than this threshold.

           Returns:
           - converged (bool): True if the state estimate has converged, False otherwise.
           """
        i = self.curr_state
        loss = np.linalg.norm(self.params.dX[:, i] - self.params.dX[:, i - 1])
        return loss < self.conv_rate


class IEKFParams:
    def __init__(self, kf):
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


class UKF:
    def __init__(self, args):
        # Initialization parameters
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        self.del_t = args.time_res
        self.state_size = args.kf_state_size
        self.curr_state = None
        self.params = UKFParams(self)  # UKF parameters
        self.traj = BaseTraj(args.run_points)
        self.epsilon = 1e-6
        self.measurement_size = 1
        self.traj.pinpoint = PinPoint(self.run_points)

    def run(self, map_data, meas):
        self._initialize_traj(meas)
        self._initialize_params()

        desc = "Estimating Trajectory with UKF"
        for i in tqdm(range(1, self.run_points), desc=desc):
            self.curr_state = i

            # Predict step
            X_sigma = self._compute_sigma_points()
            X_sigma_pred = self._predict_sigma_points(X_sigma, meas)
            self._predict_state(X_sigma_pred)
            P_pred = self._predict_covariance(X_sigma_pred)

            # Update step
            lat, lon = self._pinpoint_coordinates(meas)
            self._find_slopes(lat, lon, P_pred, map_data)  # updates SN, SE, Rfit
            h_asl_meas = meas.pos.h_asl[i]
            jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
            h_agl_meas = meas.pinpoint.range[i] * jac
            h_map_meas = RegularGridInterpolator(map_data.axis['lon'], map_data.axis['lat'], map_data.grid)(lon, lat)
            self.traj.pos.h_agl[i] = h_agl_meas

            self._calc_rc(h_agl_meas)
            self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]  # calc noise

            Z_sigma = self._unscented_transform(X_sigma_pred, map_data)
            Z_pred = self._compute_predicted_measurement(Z_sigma)
            self._update_state(X_sigma_pred, Z_sigma, Z_pred, h_asl_meas, h_agl_meas, h_map_meas, meas)
            self._update_covariance(X_sigma_pred, Z_sigma, Z_pred)

        return self

    def _predict_state(self, X_sigma_pred):
        # Predict the state estimate based on sigma points
        x_pred = np.zeros(self.state_size)
        for i in range(2 * self.state_size + 1):
            x_pred += self.params.weights_mean[i] * X_sigma_pred[:, i]
        return x_pred

    def _predict_sigma_points(self, X_sigma, meas):
        i = self.curr_state
        X_sigma_pred = np.zeros_like(X_sigma)
        for j in range(2 * self.state_size + 1):
            X_sigma_pred[0, j] = X_sigma[0, j] + (
                    X_sigma[3, j] * self.del_t + 0.5 * meas.acc.north[i - 1] * self.del_t ** 2) / meas.mpd_north[i]
            X_sigma_pred[1, j] = X_sigma[1, j] + (
                    X_sigma[4, j] * self.del_t + 0.5 * meas.acc.east[i - 1] * self.del_t ** 2) / meas.mpd_east[i]
            X_sigma_pred[2, j] = X_sigma[2, j] + (
                    X_sigma[5, j] * self.del_t + 0.5 * meas.acc.down[i - 1] * self.del_t ** 2)
            X_sigma_pred[3, j] = X_sigma[3, j] + meas.acc.north[i - 1] * self.del_t
            X_sigma_pred[4, j] = X_sigma[4, j] + meas.acc.east[i - 1] * self.del_t
            X_sigma_pred[5, j] = X_sigma[5, j] + meas.acc.down[i - 1] * self.del_t
            X_sigma_pred[6:9, j] = X_sigma[6:9, j]  # Accelerations remain the same
            X_sigma_pred[9, j] = X_sigma[9, j] + ((meas.euler.psi[i] - meas.euler.psi[i - 1]) * self.del_t)
            X_sigma_pred[10, j] = X_sigma[10, j] + ((meas.euler.theta[i] - meas.euler.theta[i - 1]) * self.del_t)
            X_sigma_pred[11, j] = X_sigma[11, j] + ((meas.euler.phi[i] - meas.euler.phi[i - 1]) * self.del_t)
        return X_sigma_pred

    def _update_state(self, X_sigma_pred, Z_sigma, Z_pred, h_asl_meas, h_agl_meas, h_map_meas, meas):
        i = self.curr_state
        Pxz = self._compute_cross_covariance(X_sigma_pred, Z_sigma, Z_pred)
        Pzz = self._compute_innovation_covariance(Z_sigma, Z_pred)
        K = Pxz @ np.linalg.pinv(Pzz)

        innovation = h_asl_meas - h_agl_meas - h_map_meas - Z_pred
        self.params.dX[:, i] = K @ innovation

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
        self.traj.pos.h_map[i] = self.traj.pos.h_asl[i] - self.traj.pos.h_agl[i]
        self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]
        self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]

    def _update_covariance(self, X_sigma_pred, Z_sigma, Z_pred):
        i = self.curr_state
        Pxz = self._compute_cross_covariance(X_sigma_pred, Z_sigma, Z_pred)
        Pzz = self._compute_innovation_covariance(Z_sigma, Z_pred)
        K = Pxz @ np.linalg.pinv(Pzz)
        self.params.P_est[:, :, i] = self._predict_covariance(X_sigma_pred) - K @ Pzz @ K.T

    def _compute_sigma_points(self):
        # Compute the sigma points for the current state
        sigma_points = np.zeros((self.state_size, 2 * self.state_size + 1))

        x = self._get_current_state_estimate()
        P = self._get_current_covariance_estimate()

        sigma_points[:, 0] = x
        # todo: choose epsilon
        sqrt_matrix = np.linalg.cholesky((self.state_size + self.params.lambda_) * P
                                         + self.epsilon * np.eye(self.state_size))

        for i in range(self.state_size):
            sigma_points[:, i + 1] = x + sqrt_matrix[:, i]
            sigma_points[:, i + 1 + self.state_size] = x - sqrt_matrix[:, i]

        return sigma_points

    def _unscented_transform(self, X_sigma_pred, map_data):
        i = self.curr_state
        Z_sigma = np.zeros((1, 2 * self.state_size + 1))
        for j in range(2 * self.state_size + 1):
            jac = cosd(X_sigma_pred[10, j]) * cosd(X_sigma_pred[11, j])
            h_agl_pred = self.traj.pinpoint.range[i] * jac
            h_map_pred = RegularGridInterpolator(map_data.axis['lon'], map_data.axis['lat'], map_data.grid)(
                X_sigma_pred[1, j],
                X_sigma_pred[0, j])
            Z_sigma[0, j] = X_sigma_pred[2, j] - h_agl_pred - h_map_pred
        return Z_sigma

    def _compute_predicted_measurement(self, Z_sigma):
        # Compute the predicted measurement
        z_pred = np.zeros(self.measurement_size)
        for i in range(2 * self.state_size + 1):
            z_pred += self.params.weights_mean[i] * Z_sigma[:, i]
        return z_pred

    def _compute_cross_covariance(self, X_sigma_pred, Z_sigma, Z_pred):
        Pxz = np.zeros((self.state_size, self.measurement_size))
        for i in range(2 * self.state_size + 1):
            Pxz += self.params.weights_covariance[i] * np.outer(X_sigma_pred[:, i] - self._predict_state(X_sigma_pred),
                                                                Z_sigma[:, i] - Z_pred)
        return Pxz

    def _compute_innovation_covariance(self, Z_sigma, Z_pred):
        Pzz = np.zeros((self.measurement_size, self.measurement_size))
        for i in range(2 * self.state_size + 1):
            Pzz += self.params.weights_covariance[i] * np.outer(Z_sigma[:, i] - Z_pred, Z_sigma[:, i] - Z_pred)
        Pzz += self.params.R[self.curr_state]
        return Pzz

    def _initialize_params(self):
        self.params = UKFParams(self)
        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2, 1, 1, 1, 1, 1, 1]), 2)
        self.params.Q = np.power(np.diag([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 1e-7)

    def _initialize_traj(self, meas):
        """
        Set initial values from measurements

        """

        self.traj.pos.lat[0] = meas.pos.lat[0]
        self.traj.pos.lon[0] = meas.pos.lon[0]
        self.traj.pos.h_asl[0] = meas.pos.h_asl[0]  # altitude
        jac = cosd(meas.euler.theta[0]) * cosd(meas.euler.phi[0])
        self.traj.pos.h_agl[0] = meas.pinpoint.range[0] * jac  # H_agl_p
        #
        self.traj.vel.north[0] = meas.vel.north[0]
        self.traj.vel.east[0] = meas.vel.east[0]
        self.traj.vel.down[0] = meas.vel.down[0]
        #
        self.traj.pinpoint.delta_north[0] = meas.pinpoint.delta_north[0]
        self.traj.pinpoint.delta_east[0] = meas.pinpoint.delta_east[0]
        self.traj.pinpoint.h_map[0] = meas.pinpoint.h_map[0]
        #
        self.traj.euler.psi[0] = meas.euler.psi[0]
        self.traj.euler.theta[0] = meas.euler.theta[0]
        self.traj.euler.phi[0] = meas.euler.phi[0]

    def _find_slopes(self, lat, lon, p_pre, map_data):
        """
        calculate the north-south and east-west slopes of terrain at a given lat lon,
        and estimate the Error of the slope model

        :param lat: Latitude at point [deg]
        :param lon: Longitude at point [deg]
        :param p_pre: A priori Error covariance matrix
        :param map_data: MapLoad instance that contain the grid and it's axes
        :return: SN, SE: slopes values , RFIT: fit Error
        """
        i = self.curr_state
        dP = 100  # distance increments in [m]
        delPmap = np.array([dP / map_data.mpd['north'][i], dP / map_data.mpd['east'][i]])  # [deg]

        # max number of points in each direction
        maxP = np.sqrt(max(p_pre[0][0], p_pre[1][1]))
        KP = 3
        NC = np.ceil(max(KP, 2 * np.ceil(KP * maxP / dP) + 1) / 2)
        idx = (NC - 1) / 2  # indices

        # create lat lon vectors according to grid indices
        pos_offset = np.arange(-idx, idx + 1)
        lat_vec, lon_vec = delPmap[0] * pos_offset + lat, delPmap[0] * pos_offset + lon
        xp, yp = np.meshgrid(lon_vec, lat_vec)
        xp, yp = xp[0, :], yp[:, 0]
        # scaling factors for slope calc
        sx2 = (dP ** 2) * 2 * NC * np.sum(np.power(np.arange(1, idx + 1), 2))
        sy2 = sx2

        # interpolate elevation data for current location
        interpolator = RegularGridInterpolator(map_data.axis['lon'], map_data.axis['lat'], map_data.grid)
        ref_elevation = interpolator(lon, lat)[0]
        grid_elevations = interpolator(xp, yp)

        # calculate slopes in x and y directions
        syh = dP * np.dot(pos_offset, grid_elevations - ref_elevation).sum()
        sxh = dP * np.dot(grid_elevations - ref_elevation, pos_offset).sum()

        self.params.SN[i], self.params.SE[i] = sxh / sx2, syh / sy2

        SN, SE = self.params.SN[i], self.params.SE[i]
        MP = (2 * idx + 1) ** 2  # number of points in the mesh grid
        In = 0
        # calculate the Error over the grid
        for step_E in np.arange(-idx, idx + 1):
            for step_N in np.arange(-idx, idx + 1):
                north = int(step_N + idx)
                east = int(step_E + idx)
                In = In + (dP * (SN * step_N + SE * step_N) - grid_elevations[east][north] + ref_elevation) ** 2
        self.params.Rfit[i] = In / (MP - 1)

    def _calc_rc(self, h_agl_meas):
        self.params.Rc[self.curr_state] = 100 + 125 * (h_agl_meas > 200) + 175 * (h_agl_meas > 760) + 600 * (
                h_agl_meas > 1000) + 500 * (
                                                  h_agl_meas > 5000) + 1500 * (h_agl_meas > 7000)

    def _predict_covariance(self, X_sigma_pred):
        x_pred = self._predict_state(X_sigma_pred)
        P_pred = np.zeros((self.state_size, self.state_size))
        for i in range(2 * self.state_size + 1):
            diff = X_sigma_pred[:, i] - x_pred
            P_pred += self.params.weights_covariance[i] * np.outer(diff, diff)
        P_pred += self.params.Q
        return P_pred

    def _get_current_state_estimate(self):
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
        return self.params.P_est[:, :, self.curr_state]

    def _pinpoint_coordinates(self, meas):
        """
        calculate current lat, lon

        :param meas:
        :return:
        """
        i = self.curr_state
        psi = meas.euler.psi[i]
        theta = meas.euler.theta[i]
        phi = meas.euler.phi[i]

        self.traj.pinpoint.delta_north[i] = meas.pinpoint.range[i] * (
                cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi))
        self.traj.pinpoint.delta_east[i] = meas.pinpoint.range[i] * (
                sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi))

        # lon - lat at pinpoint
        lat = self.traj.pos.lat[i] + self.traj.pinpoint.delta_north[i] / meas.mpd_north[i]
        lon = self.traj.pos.lon[i] + self.traj.pinpoint.delta_east[i] / meas.mpd_east[i]

        return lat, lon


class UKFParams:
    def __init__(self, ukf):
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

        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self._compute_lambda()
        self.gamma = np.sqrt(self.state_size + self.lambda_)
        self.weights_mean = self._compute_weights_mean()
        self.weights_covariance = self._compute_weights_covariance()

    def _compute_lambda(self):
        return self.alpha ** 2 * (self.state_size + self.kappa) - self.state_size

    def _compute_weights_mean(self):
        weights = np.full(2 * self.state_size + 1, 1 / (2 * (self.state_size + self.lambda_)))
        weights[0] = self.lambda_ / (self.state_size + self.lambda_)
        return weights

    def _compute_weights_covariance(self):
        weights = np.full(2 * self.state_size + 1, 1 / (2 * (self.state_size + self.lambda_)))
        weights[0] = self.lambda_ / (self.state_size + self.lambda_) + (1 - self.alpha ** 2 + self.beta)
        return weights


"""
def _update_estimate_state(self, meas):
    # Update estimated state
    i = self.curr_state

    # Update positions
    self.traj.pos.lat[i] += self.params.dX[0, i] / meas.mpd_north[i]
    self.traj.pos.lon[i] += self.params.dX[1, i] / meas.mpd_east[i]
    self.traj.pos.h_asl[i] += self.params.dX[2, i]

    # Update velocities
    self.traj.vel.north[i] += self.params.dX[3, i]
    self.traj.vel.east[i] += self.params.dX[4, i]
    self.traj.vel.down[i] += self.params.dX[5, i]

    # Update accelerations
    # Assuming the accelerations are part of the state vector and are being estimated
    self.traj.acc.north[i] += self.params.dX[6, i]
    self.traj.acc.east[i] += self.params.dX[7, i]
    self.traj.acc.down[i] += self.params.dX[8, i]

    # Update Euler angles
    self.traj.euler.psi[i] += self.params.dX[9, i]
    self.traj.euler.theta[i] += self.params.dX[10, i]
    self.traj.euler.phi[i] += self.params.dX[11, i]

    # Additional updates if needed
    self.traj.pos.h_map[i] = self.traj.pos.h_asl[i] - self.traj.pos.h_agl[i]  # Update height above map
    self.traj.pos.north[i] = self.traj.pos.lat[i] * meas.mpd_north[i]  # Convert latitude to north position
    self.traj.pos.east[i] = self.traj.pos.lon[i] * meas.mpd_east[i]  # Convert longitude to east position

"""


class baseKF:
    pass
