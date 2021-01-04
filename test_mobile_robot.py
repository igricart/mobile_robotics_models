from rotation import Rotation
from mobile_robot import MobileRobot
import numpy as np
import unittest

class Test(unittest.TestCase):
    def test_mobile_robot_constructor(self):
        print("Test Constructor")
        mr = MobileRobot(np.array([0, 0, np.deg2rad(90)]), 0.1, 1, 1)
        np.testing.assert_array_equal([0, 0, np.deg2rad(90), 0.1, 1, 1], mr.states)
        vel_input = np.array([4, 2])
        mr.update(vel_input)

    # def test_rotation_from_odom_to_reference_frame(self):
        
    #     # map motion in global frame (_I) to local frame (_r)
    #     # xi_r = R(theta) * xi_I
    #     initial_condition = np.array([0, 0, 0])
    #     mr = MobileRobot(initial_condition, 0.1)
    #     mr.update(np.array([0,0,1]))
    #     mr.update(np.array([0,0,1]))
    #     print(mr.states)
    #     mr.update(np.array([0,0,1]))
    #     mr.update(np.array([0,0,1]))
    #     print(mr.states)
    #     #self.assertAlmostEqual(result, expected_result)

    # def test_mobile_robot_rotation(self):
    #     initial_condition = np.array([0, 0, 0])
    #     mr = MobileRobot(initial_condition, 0.1, 1, 1)
    #     mr.update(np.array([0,0,1]))
    #     mr.update(np.array([0,0,1]))
    #     print(mr.states)
    #     mr.update(np.array([0,0,1]))
    #     mr.update(np.array([0,0,1]))
    #     print(mr.states)


unittest.main()