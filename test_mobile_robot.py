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

    def test_mobile_robot_update_with_integral_method_wheels_opposite_direction(self):
        print("Test wheels running in opposite direction")
        # (v_r - v_l) * t / ( distance_between_wheel_and_rotation_point * 2) = 2 * pi => distance_between_wheel_and_rotation_point = 10 / pi
        mr = MobileRobot(np.array([0, 0, 0]), 0.1, 1, 2 / np.pi)
        vel_input = np.array([2, -2])
        for i in range(10):
            i += 1
            output = mr.update_method_integral(vel_input)
            np.testing.assert_array_equal(output[0], 0)
            np.testing.assert_array_equal(output[1], 0)
            np.testing.assert_array_almost_equal(output[2], i*np.pi/10, decimal=2)
            pass

    def test_mobile_robot_update_with_integral_method_move_around_initial_pose_is_zero(self):
        print("Move around with zero initial pose")
        # (v_r - v_l) * t / ( distance_between_wheel_and_rotation_point * 2) = 2 * pi => distance_between_wheel_and_rotation_point = 10 / pi
        mr = MobileRobot(np.array([0, 0, 0]), 0.2, 1, 2 / np.pi)
        vel_input = np.array([3, 1])
        result_x = []
        result_y = []
        result_theta = []
        result_time = []
        for i in range(10):
            i += 1
            output = mr.update_method_integral(vel_input)
            result_x.append(output[0])
            result_y.append(output[1])
            result_theta.append(output[2])
            result_time.append(output[3])
            print(output)
            pass

    def test_mobile_robot_update_with_integral_method_initial_pose_different_than_zero(self):
        print("Move around with zero initial pose")
        # (v_r - v_l) * t / ( distance_between_wheel_and_rotation_point * 2) = 2 * pi => distance_between_wheel_and_rotation_point = 10 / pi
        mr = MobileRobot(np.array([0, 0, 0]), 0.2, 1, 2 / np.pi)
        vel_input = np.array([3, 1])
        result_x = []
        result_y = []
        result_theta = []
        result_time = []
        for i in range(10):
            i += 1
            output = mr.update_method_integral(vel_input)
            result_x.append(output[0])
            result_y.append(output[1])
            result_theta.append(output[2])
            result_time.append(output[3])
            print(output)
            pass
        
unittest.main()