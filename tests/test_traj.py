import unittest
from ..src.pinpoint_calc import PinPoint
from ..src.data_loaders import Map, set_settings
from ..src.create_traj import CreateTraj
from ..src.noise_traj import NoiseTraj

class TestCreateTraj(unittest.TestCase):

    def test_linear_trajectory(self):
        args = set_settings()
        map_data = Map(args).load()
        create_traj = CreateTraj(args)
        result = create_traj.linear(map_data)


        self.assertEqual(result.pos.lat[0], expected_latitude)
        self.assertEqual(result.pos.lon[0], expected_longitude)



if __name__ == '__main__':
    unittest.main()
