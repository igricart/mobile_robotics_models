from rotation import Rotation
from mobile_robot import MobileRobot
import numpy as np
import unittest

class TestLinearAlgebra(unittest.TestCase):
    def test_rotatio_Z_2D(self):
        theta = np.deg2rad(90)
        rot_obj = Rotation(theta)
        result = rot_obj.RotZ2D
        expected_result = np.array([[  0,   1],
                                    [ -1,   0]])
        np.testing.assert_array_almost_equal(result, expected_result)
        #self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-05, atol=1e-08))

    def test_rotation_from_point_in_odom_frame_to_reference_frame(self):
        initial_point = np.array([[1], [1], [1]])
        rot90 = Rotation(np.deg2rad(90))

        # Expected result with Rotation in X
        result_x = np.matmul(rot90.RotX3D, initial_point)
        expected_result = np.array([[1], [1], [-1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with Rotation in Y
        result_x = np.matmul(rot90.RotY3D, initial_point)
        expected_result = np.array([[-1], [1], [1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with Rotation in Z
        # The point [1;1;1] when represented in a frame rotated by 90 degrees is at position [1;-1;1]
        result_z = np.matmul(rot90.RotZ3D, initial_point)
        expected_result = np.array([[1], [-1], [1]])
        np.testing.assert_array_almost_equal(result_z, expected_result)

    def test_rotation_from_point_in_reference_frame_to_odom_frame(self):
        initial_point = np.array([[1], [1], [1]])
        rot90 = Rotation(np.deg2rad(90))

        # Expected result with transpose Rotation in X
        result_x = np.matmul(rot90.RotX3D.transpose(), initial_point)
        expected_result = np.array([[1], [-1], [1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with transpose Rotation in Y
        result_x = np.matmul(rot90.RotY3D.transpose(), initial_point)
        expected_result = np.array([[1], [1], [-1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with transpose Rotation in Z
        # The point [-1;1;1] when represented in a frame rotated by 90 degrees in Z is at position [1;1;1]
        result_z = np.matmul(rot90.RotZ3D.transpose(), initial_point)
        expected_result = np.array([[-1], [1], [1]])
        np.testing.assert_array_almost_equal(result_z, expected_result)


unittest.main()