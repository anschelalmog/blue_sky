import numpy as np
from .utils import cosd, sind
from .base_classes import BaseTraj, IEKFParams
from scipy.interpolate import interp2d
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt


class IteratedExtendedKF:
    """
        simulating an iterated extended kalman filter
        the number of loops and convergence rate declared in input
    """

    def __init__(self, args, map_data, meas_traj):
        self.delT = args.time_res
        self.epsilon = args.iekf_conv_rate
        self.run_points = args.run_points
        self.max_iter = args.iekf_iters
        self.state_size = args.kf_state_size
        self.est_traj = BaseTraj(self)
        self.params = IEKFParams(self)
        self.current_state = None
        # set Kalman parameters, set first estimation to measures
        self.initialize_params(args, meas_traj)

        # run IEKF
        self.run(map_data, meas_traj)

        self.est_traj.pos_North = self.est_traj.Lat * meas_traj.pos_mpd_N
        self.est_traj.pos_East = self.est_traj.Lon * meas_traj.pos_mpd_E

    def run(self, map_data, meas_traj):
        """
        main process of iterated extended kalman filter
        :param
        epsilon - convergence rate
        max_iter - maximum number of iterations per prediction
        :param map_data:
        :param meas_traj:
        """
        desc = "Iterated Extended Kalman Filter calculations"
        kalman_filter_process = tqdm(range(1, self.run_points), desc=desc, position=0)

        # main estimation loop
        for i in kalman_filter_process:
            self.current_state = i

            # prediction
            self.predict_state(meas_traj)  # update estimation according to prediction
            p_cov = self.predict_covariance()

            update_loop = tqdm(range(self.max_iter + 1), desc="_update_IEKF_", leave=False, position=1)
            # IEKF inner loop, estimation
            for _ in update_loop:
                # lat - lon at pinpoint
                lat, lon = self.update_location(meas_traj)

                # find slopes slopes and and errors
                interpolator = interp2d(map_data.Lat, map_data.Lon, map_data.map_grid)
                self.slopes_at_point(lat, lon, p_cov, map_data)
                # self.calc_slopes(lat, lon, p_cov, map_data)
                # fixme: returns measured ground and not asl
                self.est_traj.H_asl[i] = interpolator(lat, lon)
                # ic(self.est_traj.H_asl[i])
                # ic(self.params.SE[i])
                # ic(self.params.SN[i])

                H_agl_meas = meas_traj.R_pinpoint[i] * cosd(meas_traj.euler_Theta[i]) * cosd(meas_traj.euler_Phi[i])

                # Noise calculation
                # calc noise
                self.params.Rc[i] = self.rc(H_agl_meas)
                self.params.R[i] = self.params.Rc[i] + self.params.Rfit[i]

                # Measurements matrix update
                self.params.H[:3, i] = [-self.params.SN[i], -self.params.SE[i], -1]

                # Kalman Gain
                self.params.K[:, i] = self.calculate_gain()

                # Measurement Model
                # estimated height above sea - measured height above sea - measured map height
                self.params.Z[i] = self.est_traj.H_asl[i] - H_agl_meas - meas_traj.H_map_pinpoint[i]
                # ic(i)
                # ic(self.params.Z[i])

                self.params.P_est[:, :, i] = self.estimate_covariance()
                self.params.dX[:, i] = self.params.K[:, i] * self.params.Z[i]
                # ic(self.est_traj.H_asl[i])

                if self.converged():
                    break

            self.estimation_update(meas_traj)

    def estimation_update(self, meas_traj):
        i = self.current_state
        self.est_traj.Lat[i] = self.est_traj.Lat[i - 1] - self.params.dX[0, i] / meas_traj.pos_mpd_N[i - 1]
        self.est_traj.Lon[i] = self.est_traj.Lon[i - 1] - self.params.dX[1, i] / meas_traj.pos_mpd_E[i - 1]
        self.est_traj.H_asl[i] = self.est_traj.H_asl[i - 1] - self.params.dX[2, i]
        self.est_traj.vel_North[i] = self.est_traj.vel_North[i - 1] + self.params.dX[3, i]
        self.est_traj.vel_East[i] = self.est_traj.vel_East[i - 1] + self.params.dX[4, i]
        self.est_traj.vel_Down[i] = self.est_traj.vel_Down[i - 1] + self.params.dX[5, i]

    def initialize_params(self, args, meas_traj):
        # first estimation of position and velocity is taking from the measurements
        self.est_traj.vel_North[0] = meas_traj.vel_North[0]
        self.est_traj.vel_East[0] = meas_traj.vel_East[0]
        self.est_traj.vel_Down[0] = meas_traj.vel_Down[0]
        self.est_traj.pos_North[0], self.est_traj.pos_East[0] = meas_traj.pos_North[0], meas_traj.pos_East[0]
        self.est_traj.H_asl[0] = meas_traj.H_asl[0]
        self.est_traj.Lat[0], self.est_traj.Lon[0] = meas_traj.Lat[0], meas_traj.Lon[0]

        jac = cosd(meas_traj.euler_Theta[1]) * cosd(meas_traj.euler_Phi[1])
        self.params.dN_p[1] = meas_traj.dN[1]
        self.params.dE_p[1] = meas_traj.dE[1]
        self.params.H_agl_p[1] = meas_traj.R_pinpoint[1] * jac
        self.params.H_map = meas_traj.H_map_pinpoint[1]

        # Dynamic Equation: linear motion
        # dX_k + 1 = Phi_k + 1 | k * dX_k + W_k + 1
        self.params.Phi = np.eye(args.kf_state_size)
        self.params.Phi[0][3] = args.time_res
        self.params.Phi[1][4] = args.time_res
        self.params.Phi[2][5] = args.time_res

        self.params.P_est[:, :, 0] = args.kf_P_est0
        self.params.Q = args.kf_Q

    def update_location(self, meas_traj):
        """
        Calculate PinPoint coordinates
        :param meas_traj:
        :return:
        """
        i = self.current_state
        psi = meas_traj.euler_Psi[i]
        theta = meas_traj.euler_Theta[i]
        phi = meas_traj.euler_Phi[i]

        jac_to_north = cosd(psi) * cosd(theta) * cosd(phi) + sind(psi) * sind(phi)
        jac_to_east = sind(psi) * sind(theta) * cosd(phi) - cosd(psi) * sind(phi)

        self.params.dN_p[i] = meas_traj.R_pinpoint[i] * jac_to_north
        self.params.dE_p[i] = meas_traj.R_pinpoint[i] * jac_to_east

        lat = self.est_traj.Lat[i] + self.params.dN_p[i] / meas_traj.pos_mpd_N[i]
        lon = self.est_traj.Lon[i] + self.params.dE_p[i] / meas_traj.pos_mpd_E[i]

        return lat, lon

    def predict_covariance(self):
        """
        Compute prior Error covariance matrix
        :return: P(t|t-1) = F * P(t-1|t-1) * F^T + Q
        """
        i = self.current_state
        return self.params.Phi @ self.params.P_est[:, :, i - 1] @ self.params.Phi.T + self.params.Q

    def predict_state(self, meas_traj):
        """
        compute estimated velocities and positions based on previous and current measurements
        :param meas_traj: measurement trajectory
        :return: updating class
        """
        i = self.current_state
        # self.est_traj.vel_North[i] = self.est_traj.vel_North[i - 1] + (
        #             meas_traj.vel_North[i] - meas_traj.vel_North[i - 1])
        self.est_traj.vel_East[i] = self.est_traj.vel_East[i - 1] + (meas_traj.vel_East[i] - meas_traj.vel_East[i - 1])
        self.est_traj.vel_Down[i] = self.est_traj.vel_Down[i - 1] + (meas_traj.vel_Down[i] - meas_traj.vel_Down[i - 1])

        self.est_traj.Lat[i] = self.est_traj.Lat[i - 1] + (meas_traj.vel_North[i] / meas_traj.pos_mpd_N[i]) * self.delT
        self.est_traj.Lon[i] = self.est_traj.Lon[i - 1] + (meas_traj.vel_East[i] / meas_traj.pos_mpd_E[i]) * self.delT
        self.est_traj.H_asl[i] = self.est_traj.H_asl[i - 1] + meas_traj.H_asl[i] - meas_traj.H_asl[i - 1]

    def slopes_at_point(self, lat, lon, P_pre_cov, map_data):
        """
        calculate the north-south and east-west slopes of terrain at a given lat lon,
        and estimate the Error of the slope model
        :param lat: Latitude at point [deg]
        :param lon: Longitude at point [deg]
        :param P_pre_cov: A priori Error covariance matrix
        :param map_data: MapLoad instance that contain the grid and it's axes
        :return: SN, SE: slopes values , RFIT: fit Error
        """
        i = self.current_state
        dP = 100  # distance increments in [m]
        delPmap = np.array([dP / map_data.mpd_N[i], dP / map_data.mpd_E[i]])  # [deg]

        # max number of points in each direction
        maxP = np.sqrt(max(P_pre_cov[0][0], P_pre_cov[1][1]))
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
        interpolator = interp2d(map_data.Lon, map_data.Lat, map_data.map_grid)
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

    def converged(self):
        i = self.current_state
        return np.linalg.norm(self.params.dX[:, i] - self.params.dX[:, i - 1]) <= self.epsilon

    def calculate_gain(self):
        i = self.current_state
        H = self.params.H[:, i]
        P = self.params.P_est[:, :, i]
        R = self.params.R[i]
        return P @ H.T / (H @ P @ H.T + R)

    def estimate_covariance(self):
        """
        compute covariance
        :return: current covariance
        """
        idx = self.current_state
        Eye = np.eye(self.state_size)
        K = self.params.K[:, idx].reshape(-1, 1)
        H = np.transpose(self.params.H[:, idx].reshape(-1, 1))
        R = np.array([[self.params.R[idx]]])
        P = self.params.P_est[:, :, idx]

        return (Eye - K @ H) @ P @ (Eye - K @ H).T + K @ R @ K.T

    def rc(self, H_agl_meas):
        return 100 + 125 * (H_agl_meas > 200) + 175 * (H_agl_meas > 760) + 600 * (H_agl_meas > 1000) + 500 * (
                H_agl_meas > 5000) + 1500 * (H_agl_meas > 7000)


class UnscentedKF:
    # def __init__(self, args, meas_traj, map_data):
    def __init__(self):
        pass
    #     self.delT = args.time_res
    #     self.max_iter = args.iekf_iters
    #     self.est_traj = BaseTraj(args)
    #     se   clf.params = UKFParams(args)
    #
    #     # set params
    #     self.initialize_params(args, meas_traj)
    #     # run UKF
    #     self.run(args, map_data, meas_traj)
    #
    # def run(self, args, map_data, meas_traj):
    #     pass
    #
    # def f(x, dt):
    #     F = np.array([
    #          [1, 0, 0, dt, 0, 0],
    #          [0, 1, 0, 0, dt, 0],
    #          [0, 0, 1, 0, 0, dt],
    #          [0, 0, 0, 1, 0, 0],
    #          [0, 0, 0, 0, 1, 0],
    #          [0, 0, 0, 0, 0, 1],
    #      ])
    #     return np.dot(F, x)
    #
    #
    # def h(x):
    #     return np.array([x[0], x[1], x[2]])
    #
    #
    # def prediction(x_aug, P_aug, wm, wc, dt, Q):
    #     sqrt_P_aug = np.linalg.cholesky(P_aug)
    #     sigma_pts = np.hstack([x_aug[:, None], x_aug[:, None] + np.sqrt(n_aug + lambda_) * sqrt_P_aug,
    #                            x_aug[:None] - np.sqrt(n_aug + lambda_) * sqrt_P_aug])
    #     sigma_pts_pred = np.apply_along_axis(f, 0, sigma_pts[:n, :], dt)
    #     x_pred = np.dot(wm, sigma_pts_pred.T)
    #
    #     P_pred = np.zeros((n, n))
    #     for i in range(2 * n_aug + 1):
    #         diff = sigma_pts_pred[:, i] - x_pred
    #         P_pred += wc[i] * np.outer(diff, diff)
    #     P_pred += Q
    #     return x_pred, P_pred, sigma_pts_pred
    #
    #
    # def update(x_pred, P_pred, sigma_points_pred, wm, wc, y, x_aug):
    #     sigma_points_meas = np.apply_along_axis(h, 0, sigma_points_pred)
    #     z_pred = np.dot(wm, sigma_points_meas.T)
    #
    #     S = np.zeros((m, m))
    #     for i in range(2 * n_aug + 1):
    #         diff = sigma_points_meas[:, i] - z_pred
    #         S += wc[i] * np.outer(diff, diff)
    #     S += np.diag(x_aug[-m:])
    #
    #     C = np.zeros((n, m))
    #     for i in range(2 * n_aug + 1):
    #         diff_x = sigma_points_pred[:, i] - z_pred
    #         diff_z = sigma_points_meas[:, i] - z_pred
    #         C += wc[i] * np.outer(diff_x, diff_z)
    #
    #     K = np.linalg.solve(S, C.T).T
    #     x_updated = x_pred + np.dot(K, (y - z_pred))
    #     P_updated = P_pred - np.dot((np.dot(K, S), K.T))
    #
    #     return x_updated, P_updated
    #
    #
    # n = 6
    # m = 3
    # x_real = np.array([0, 0, 0, 1, 1, 1])
    # x_aug = np.array([0, 0, 0, 1, 1, 1, 0.1, 0.1, 0.1])
    # n_aug = n + m
    # P_aug = np.eye(n_aug)
    # Q = np.eye(n) * 0.1
    # Q_aug = np.diag(np.hstack([np.diag(Q), np.zeros(m)]))
    #
    # alpha = 0.1
    # beta = 2.0
    # kappa = 0
    # lambda_ = alpha ** 2 * (n_aug + kappa) - n_aug
    #
    # dt = 0.1
    #
    # real_states = []
    # meas_states = []
    # est_states = []
    #
    # wm = np.zeros(2 * n_aug + 1)
    # wc = np.zeros(2 * n_aug + 1)
    # wm[0] = lambda_ / (n_aug + lambda_)
    # wc[0] = lambda_ / (n_aug + lambda_) + (1 - alpha ** 2 + beta)
    # wm[1:] = wc[1:] = 1 / (2 * (n_aug + lambda_))
    #
    # for _ in range(100):
    #     process_noise_real = np.random.normal(0, np.sqrt(0.1), n)
    #     x_real = f(x_real, dt) + process_noise_real
    #     x_pred, P_pred, sigma_points_pred = prediction(x_aug, P_aug, wm, wc, dt, Q)
    #     measurement_noise = np.random.normal(0, np.sqrt(x_aug[-m:]), m)
    #     y = h(x_real) + measurement_noise
    #
    #     x_updated, P_updated = update(x_pred, P_pred, sigma_points_pred, wm, wc, y, x_aug)
    #
    #     x_aug[:n] = x_updated
    #     P_aug[:n, :n] = P_updated
    #
    #     innovation = (y - h(x_updated))
    #     x_aug[-m:] += dt * (innovation ** 2 - x_aug[-m:])
    #
    #     real_states.append(x_real[:m])
    #     meas_states.append(y)
    #     est_states.append(x_aug[:m])
