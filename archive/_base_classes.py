"""
    this modules contain subclasses for later use
"""
import numpy as np
import matplotlib.pyplot as plt


class BaseTraj:
    def __init__(self, args):
        # velocity stats
        self.vel_North = np.zeros(args.run_points)
        self.vel_East = np.zeros(args.run_points)
        self.vel_Down = np.zeros(args.run_points)
        # position stats
        self.H_asl = np.zeros(args.run_points)
        self.H_agl = np.zeros(args.run_points)
        self.pos_North = np.zeros(args.run_points)
        self.pos_East = np.zeros(args.run_points)
        self.Lat = np.zeros(args.run_points)
        self.Lon = np.zeros(args.run_points)

        # attitude stats
        # self.eul_Psi = np.zeros(args.run_points)
        # self.eul_Theta = np.zeros(args.run_points)
        # self.eul_Phi = np.zeros(args.run_points)


class IEKFParams:
    def __init__(self, args):
        self.P_est = np.zeros((args.state_size, args.state_size, args.run_points))
        self.Q = np.zeros(args.state_size)
        self.Phi = np.zeros(args.state_size)
        self.H = np.zeros((args.state_size, args.run_points))  # measurement vector
        self.R = np.zeros(args.run_points)
        self.Rc = np.zeros(args.run_points)
        self.Rfit = np.zeros(args.run_points)
        self.K = np.zeros((args.state_size, args.run_points))
        self.dN_p = np.zeros(args.run_points)
        self.dE_p = np.zeros(args.run_points)
        self.dX = np.zeros((args.state_size, args.run_points))
        self.Z = np.zeros(args.run_points)
        self.H_agl_p = np.zeros(args.run_points)
        self.H_map = np.zeros(args.run_points)
        self.SN = np.zeros(args.run_points)
        self.SE = np.zeros(args.run_points)



class KalmanFilterParameters:
    def __init__(self, args):
        self.P_est = np.zeros((args.state_size, args.state_size, args.run_points))
        self.Q = np.zeros(args.state_size)
        self.Phi = np.zeros(args.state_size)
        self.H = np.zeros((args.state_size, args.run_points))  # measurement vector
        self.R = np.zeros(args.run_points)
        self.Rc = np.zeros(args.run_points)
        self.Rfit = np.zeros(args.run_points)
        self.K = np.zeros((args.state_size, args.run_points))
        self.dN_p = np.zeros(args.run_points)
        self.dE_p = np.zeros(args.run_points)
        self.dX = np.zeros((args.state_size, args.run_points))
        self.Z = np.zeros(args.run_points)  # measurement vector
        self.H_agl_p = np.zeros(args.run_points)
        self.H_map = np.zeros(args.run_points)
        self.SN = np.zeros(args.run_points)
        self.SE = np.zeros(args.run_points)



class UKFParams:
    def __init__(self, args):
        self.alpha = args.ukf_alpha
        self.beta = args.ukf_beta
        self.kappa = args.ukf_alpha
        self.P_est = np.zeros((args.state_size + 3, args.state_size + 3, args.run_points))
        self.Q = np.zeros(args.state_size)
        self.Q_est = np
        self.Phi = np.zeros(args.state_size)


class Errors(BaseTraj):
    def __init__(self, args, meas_traj, estimation_results, true_traj=None):
        super().__init__(args)

        self.true = true_traj
        self.meas = meas_traj
        self.est = estimation_results

        covariances = ['pos_north', 'pos_east', 'pos_down', 'vel_north', 'vel_east', 'vel_down']
        # 'eul_psi', 'eul_theta', 'eul_phi']
        for cov in range(len(covariances)):
            setattr(self, f'cov_{covariances[cov]}', estimation_results.params.P_est[cov - 1, cov - 1, :])

        setattr(self, 'cov_pos_north_est', estimation_results.params.P_est[1, 2, :])
        setattr(self, 'cov_pos_east_north', estimation_results.params.P_est[2, 1, :])

        self.pos_North = self.meas.pos_North - self.est.est_traj.pos_North
        self.pos_East = self.meas.pos_East - self.est.est_traj.pos_East
        self.pos_alt = self.true.H_asl - self.est.est_traj.H_asl
        self.vel_North = self.true.vel_North - self.est.est_traj.vel_North
        self.vel_East = self.true.vel_East - self.est.est_traj.vel_East
        self.vel_alt = self.true.vel_Down - self.est.est_traj.vel_Down
