from rotation import LA
from mobile_robot import MobileRobot
import numpy as np
import unittest
import matplotlib.pyplot as plt

class Test(unittest.TestCase):
    def test_mobile_robot_constructor(self):
        print("Test Constructor")
        mr = MobileRobot(np.array([0, 0, np.deg2rad(90)]), 0.1, 1, 1)
        #np.testing.assert_array_equal([0, 0, np.deg2rad(90), 0.1, 1, 1], mr.states)
        vel_input = np.array([4, 2])
        mr.update(vel_input)

    def test_state(self):
        print("Test method states")
        mr = MobileRobot(np.array([0, 0, np.deg2rad(90)]), 0.1, 1, 1)
        print(mr.states)

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
            np.testing.assert_array_almost_equal(output[2], (i-1)*np.pi/10, decimal=2)
            pass

    def test_mobile_robot_update_with_integral_method_move_around_initial_pose_is_zero(self):
        print("Move around with zero initial pose")
        # (v_r - v_l) * t / ( distance_between_wheel_and_rotation_point * 2) = 2 * pi => distance_between_wheel_and_rotation_point = 10 / pi
        resolution = 3
        period = 1
        vel_input = np.array([3, 1])
        distance_between_wheel_and_rotation_point = (vel_input[0] - vel_input[1]) * period / (4 * np.pi)
        dt = period / resolution
        mr = MobileRobot(np.array([0, 0, 0]), dt, 1, distance_between_wheel_and_rotation_point)
        result_x = []
        result_y = []
        result_theta = []
        result_time = []
        i = 0
        print ("_x | _y | _theta | _cumulative_dt | _dt")
        for i in range(resolution + 1):
            output = mr.update_method_integral(vel_input)
            result_x.append(output[0])
            result_y.append(output[1])
            result_theta.append(output[2])
            result_time.append(output[3])
            i += 1
            print(output)
            pass

        figure = plt.figure()
        plt.setp(figure, animated=True)
        plt.subplot(121)
        plt.plot(result_x, result_y)
        plt.xlabel('X value')
        plt.subplot(122)
        plt.plot(result_theta, result_time)
        plt.ylabel('Theta value')
        plt.show()

    def test_mobile_robot_update_with_integral_method_move_around_initial_pose_is_zero_known_case(self):
        print("Move around with zero initial pose in a known case")
        # # (v_r - v_l) * t / ( distance_between_wheel_and_rotation_point * 2) = 2 * pi => distance_between_wheel_and_rotation_point = 10 / pi
        # mr = MobileRobot(np.array([0, 0, 0]), 0.05, 1, 0.1)
        # # Expected result for a circle of radius one done in a period of 2 seconds
        # expected_result_1 = np.array([0, 0, 0, 0])
        # expected_result_2 = np.array([-1, 1, np.pi * 0.5, 0.5])
        # expected_result_3 = np.array([-2, 0, np.pi * 1, 1])
        # expected_result_4 = np.array([-1, -1, np.pi * 1.5, 1.5])
        # expected_result_5 = np.array([0, 0, 0, 2])
        # vel_input = np.array([np.pi * 1.1, np.pi * 0.9])
        # print ("_x | _y | _theta | _dt | _dt")
        # for i in range(40):
        #     i += 1
        #     output = mr.update_method_integral(vel_input)
        #     print(output)
        #     pass

    def test_mobile_robot_update_with_integral_method_initial_pose_different_than_zero(self):
        print("Move around with non-zero initial pose")
        print("Move around with zero initial pose")
        # (v_r - v_l) * t / ( distance_between_wheel_and_rotation_point * 2) = 2 * pi => distance_between_wheel_and_rotation_point = 10 / pi
        resolution = 20
        period = 1
        vel_input = np.array([3, 1])
        distance_between_wheel_and_rotation_point = (vel_input[0] - vel_input[1]) * period / (4 * np.pi)
        dt = period / resolution
        mr = MobileRobot(np.array([1, 1, np.deg2rad(45)]), dt, 1, distance_between_wheel_and_rotation_point)
        result_x = []
        result_y = []
        result_theta = []
        result_time = []
        i = 0
        print ("_x | _y | _theta | _cumulative_dt | _dt")
        for i in range(resolution + 1):
            output = mr.update_method_integral(vel_input)
            result_x.append(output[0])
            result_y.append(output[1])
            result_theta.append(output[2])
            result_time.append(output[3])
            i += 1
            print(output)
            pass

        figure = plt.figure()
        plt.setp(figure, animated=True)
        plt.subplot(121)
        plt.plot(result_x, result_y)
        plt.xlabel('X value')
        plt.subplot(122)
        plt.plot(result_time, result_theta)
        plt.ylabel('Theta value')
        plt.show()

unittest.main()