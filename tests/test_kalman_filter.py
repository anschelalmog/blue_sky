import unittest
import numpy as np
from ..main import main

class TrajectoryComparisonTests(unittest.TestCase):
    def test_trajectory_psi_angles(self):
        # Run the simulation with psi_angle set to 45
        latitudes_45 = main

        # Run the simulation with psi_angle set to 0
        latitudes_0 = main(psi_angle=0)

        # Compare the latitude position vectors from the two runs
        self.assertTrue(np.allclose(latitudes_45, latitudes_0, rtol=1e-5, atol=1e-8),
                        "The latitude vectors should be similar within a tolerance")

if __name__ == '__main__':
    unittest.main()
