import unittest
import numpy as np
from modules.data_loaders import Map
from modules.create_traj import CreateTraj
from modules.noise_traj import NoiseTraj
from modules.estimators import IEKF, UKF
from modules.outputs_utils import Errors, Covariances, plot_results, print_log


class TestIEKF(unittest.TestCase):
    def setUp(self):
        pass
        # self.iekf = IEKF(map_data, args)
        # self.iekf.curr_state = 0
        # self.iekf._initialize_params()

        # self.meas = NoiseTraj()
        # self.meas.vel.north[0] = 1
        # self.meas.vel.east[0] = 0
        # self.meas.vel.down[0] = 0
        # self.meas.euler.psi[0] = 0
        # self.meas.euler.theta[0] = 0
        # self.meas.euler.phi[0] = 0
        # self.meas.pos.h_asl[0] = 100

    def test_predict_state(self):
        pass
        # self.iekf.traj.vel.north[0] = 1
        # self.iekf.traj.vel.east[0] = 0
        # self.iekf.traj.vel.down[0] = 0
        # self.iekf.traj.pos.h_asl[0] = 100
        #

        # self.iekf._predict_state(self.meas)
        #

        # self.assertEqual(self.iekf.traj.vel.north[1], 1)
        # self.assertEqual(self.iekf.traj.vel.east[1], 0)
        # self.assertEqual(self.iekf.traj.vel.down[1], 0)
        # self.assertEqual(self.iekf.traj.pos.h_asl[1], 100)


if __name__ == '__main__':
    unittest.main()
