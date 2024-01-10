import numpy as np
from scipy.interpolate import interp2d
from tqdm import tqdm
import matplotlib.pyplot as plt
from icecream import ic

from modules.utils import cosd, sind
from modules.base_traj import BaseTraj
from modules.pinpoint_calc import PinPoint


def jac_north(psi, theta, phi):
    return cosd(psi) * sind(theta) * cosd(phi) + sind(psi) * sind(phi)


def jac_east(psi, theta, phi):
    return sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi)


#TODO: check where does the state updates by the euler angels
class IEKF:
    def __init__(self, args):
        self.run_points = args.run_points
        self.time_vec = args.time_vec
        self.del_t = args.time_res
        self.state_size = args.kf_state_size
        self.curr_state = None
        self.max_iter = None
        self.conv_rate = None
        self.traj = BaseTraj(self.run_points)
        self.traj.pinpoint = PinPoint(self.run_points)
        self.params = IEKFParams(self)


    def _initialize_traj(self, meas):
        # Set initial values from meas
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

    def _initialize_params(self):
        self.params = IEKFParams(self)

        self.params.P_est[:, :, 0] = np.power(np.diag([200, 200, 30, 2, 2, 2]), 2)
        self.params.Q = np.power(np.diag([0, 0, 0, 1, 1, 1]), 1e-6)
        # Dynamic Equation: linear motion
        # dX_k + 1 = Phi_k + 1 | k * dX_k + W_k + 1
        self.params.Phi = np.eye(self.state_size)
        self.params.Phi[0][3] = self.del_t
        self.params.Phi[1][4] = self.del_t
        self.params.Phi[2][5] = self.del_t

    def run(self, map_data, meas):
        self._initialize_traj(meas)

        self._initialize_params()

        desc = "Estimating Trajectory with IEKF"

        for i in tqdm(range(1, self.run_points), desc=desc):
            self.curr_state = i

            "-Prediction step"
            self._predict_state(meas)
            p_pre = self._predict_covariance()

            "-Measurement update"
            lat, lon = self._pinpoint_coordinates(meas)
            self._find_slopes(lat, lon, p_pre, map_data)  # updates SN, SE, Rfit
            h_asl_meas = self.traj.pos.h_asl[i]  # meas.pos.h_asl[i]
            jac = cosd(meas.euler.theta[i]) * cosd(meas.euler.phi[i])
            h_agl_meas = meas.pinpoint.range[i] * jac
            h_map_meas = interp2d(map_data.ax_lon, map_data.ax_lat, map_data.grid)(lon, lat)
            self.traj.pos.h_agl[i] = h_agl_meas

            self.params.H[:3, i] = [-self.params.SN[i], -self.params.SE[i], -1]  # Update observation matrix
            self._calc_rc(h_agl_meas)
            self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]  # calc noise

            "-Estimation step"
            self._compute_gain(p_pre)
            # Measurement Model
            self.params.Z[i] = h_asl_meas - h_agl_meas - h_map_meas
            self._estimate_covariance(p_pre)

            self.params.dX[:, i] = self.params.K[:, i] * self.params.Z[i]
            self._update_estimate_state(meas)

        return self

    def _predict_state(self, meas):
        """
        compute estimated velocities and positions based on previous and current measurements

        x_hat(t|t-1) = F * x_hat(t-1|t-1) + B * u(t)
        :param meas: measurement trajectory
        :return: updating class
        """
        i = self.curr_state
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
        delPmap = np.array([dP / map_data.mpd_north[i], dP / map_data.mpd_east[i]])  # [deg]

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
        interpolator = interp2d(map_data.ax_lon, map_data.ax_lat, map_data.grid)
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

    def _update_estimate_state(self, meas):
        # Update estimated state
        i = self.curr_state

        self.traj.pos.lat[i] -= self.params.dX[0, i] / meas.mpd_north[i]
        self.traj.pos.lon[i] -= self.params.dX[1, i] / meas.mpd_east[i]
        self.traj.pos.h_asl[i] -= self.params.dX[2, i]
        self.traj.vel.north[i] -= self.params.dX[3, i]
        self.traj.vel.east[i] -= self.params.dX[4, i]
        self.traj.vel.down[i] -= self.params.dX[5, i]
        self.traj.pos.h_map[i] = self.traj.pos.h_asl[i] - self.traj.pos.h_agl[i]


class IEKFParams:
    def __init__(self, kf):
        self.P_est = np.zeros((kf.state_size, kf.state_size, kf.run_points))
        self.Q = np.zeros((kf.state_size, kf.state_size))  # system noise, constant
        self.Phi = np.zeros((kf.state_size, kf.state_size))
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

    def run(self, map_data, meas):
        for i in range(self.run_points):
            self.curr_state = i

            # Predict step
            X_sigma = self._compute_sigma_points()
            X_sigma_pred = self._predict_sigma_points(X_sigma)
            self._predict_state(X_sigma_pred)
            self._predict_covariance(X_sigma_pred)

            # Update step
            Z_sigma = self._unscented_transform(X_sigma_pred)
            Z_pred = self._compute_predicted_measurement(Z_sigma)
            self._update_state(Z_sigma, Z_pred, meas)
            self._update_covariance(X_sigma_pred, Z_sigma, Z_pred)

    def _predict_state(self, X_sigma_pred):
        # Predict the state estimate based on sigma points
        x_pred = np.zeros(self.state_size)
        for i in range(2 * self.state_size + 1):
            x_pred += self.params.weights_mean[i] * X_sigma_pred[:, i]
        return x_pred

    def _predict_sigma_points(self, X_sigma):
        # Propagate each sigma point through the process model
        # Assuming a linear process model for demonstration purposes
        X_sigma_pred = np.zeros_like(X_sigma)
        for i in range(2 * self.state_size + 1):
            X_sigma_pred[:, i] = self._process_model(X_sigma[:, i])
        return X_sigma_pred

    def _predict_covariance(self, X_sigma_pred):
        # Predict the state covariance estimate based on sigma points
        P_pred = np.zeros((self.state_size, self.state_size))
        x_pred = self._predict_state(X_sigma_pred)
        for i in range(2 * self.state_size + 1):
            diff = X_sigma_pred[:, i] - x_pred
            P_pred += self.params.weights_covariance[i] * np.outer(diff, diff)
        return P_pred

    def _update_state(self):
        pass  # State update step

    def _update_covariance(self):
        pass  # Covariance update step

    def _compute_sigma_points(self):
        # Compute the sigma points for the current state
        sigma_points = np.zeros((self.state_size, 2 * self.state_size + 1))

        x = self._get_current_state_estimate()
        P = self._get_current_covariance_estimate()

        sigma_points[:, 0] = x
        sqrt_matrix = np.linalg.cholesky((self.state_size + self.params.lambda_) * P)

        for i in range(self.state_size):
            sigma_points[:, i + 1] = x + sqrt_matrix[:, i]
            sigma_points[:, i + 1 + self.state_size] = x - sqrt_matrix[:, i]

        return sigma_points

    def _unscented_transform(self):
        Z_sigma = np.zeros((self.measurement_size, 2 * self.state_size + 1))
        for i in range(2 * self.state_size + 1):
            Z_sigma[:, i] = self._measurement_model(X_sigma_pred[:, i])
        return Z_sigma

    def _compute_predicted_measurement(self, Z_sigma):
        # Compute the predicted measurement
        z_pred = np.zeros(self.measurement_size)
        for i in range(2 * self.state_size + 1):
            z_pred += self.params.weights_mean[i] * Z_sigma[:, i]
        return z_pred

class UKFParams:
    def __init__(self, ukf):
        # State size and run points from the UKF instance
        self.state_size = ukf.state_size
        self.run_points = ukf.run_points

        # Covariance matrices
        self.P_est = np.zeros((self.state_size, self.state_size, self.run_points))  # Estimated error covariance
        self.Q = np.zeros((self.state_size, self.state_size))  # Process noise covariance
        self.R = np.zeros(self.run_points)  # Measurement noise covariance

        # UKF  parameters # should be args
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0

        # Computed parameters
        self.lambda_ = self._compute_lambda()
        self.gamma = np.sqrt(self.state_size + self.lambda_)

        # Sigma points weights
        self.weights_mean = self._compute_weights_mean()  # Weights for the mean
        self.weights_covariance = self._compute_weights_covariance()  # Weights for the covariance

    def _compute_lambda(self):
        """
        Compute lambda using alpha, kappa, and state size.
        """
        return self.alpha ** 2 * (self.state_size + self.kappa) - self.state_size

    def _compute_weights_mean(self):
        """
        Compute weights for the sigma points mean.
        """
        weights = np.full(2 * self.state_size + 1, 1 / (2 * (self.state_size + self.lambda_)))
        weights[0] = self.lambda_ / (self.state_size + self.lambda_)
        return weights

    def _compute_weights_covariance(self):
        """
        Compute weights for the sigma points covariance.
        """
        weights = np.full(2 * self.state_size + 1, 1 / (2 * (self.state_size + self.lambda_)))
        weights[0] = self.lambda_ / (self.state_size + self.lambda_) + (1 - self.alpha ** 2 + self.beta)
        return weights

