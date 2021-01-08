from rotation import LA
from mobile_robot import MobileRobot
import numpy as np
import unittest

class TestLinearAlgebra(unittest.TestCase):
    def test_rotatio_Z_2D(self):
        rot_obj = LA
        result = rot_obj.RotZ2D(self, np.deg2rad(90))
        expected_result = np.array([[  0,   1],
                                    [ -1,   0]])
        np.testing.assert_array_almost_equal(result, expected_result)
        #self.assertTrue(numpy.allclose(result, expected_result, rtol=1e-05, atol=1e-08))

    def test_rotation_from_point_in_odom_frame_to_reference_frame(self):
        initial_point = np.array([[1], [1], [1]])
        rot90 = LA

        # Expected result with Rotation in X
        result_x = np.matmul(rot90.RotX3D(self, np.deg2rad(90)), initial_point)
        expected_result = np.array([[1], [1], [-1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with Rotation in Y
        result_x = np.matmul(rot90.RotY3D(self, np.deg2rad(90)), initial_point)
        expected_result = np.array([[-1], [1], [1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with Rotation in Z
        # The point [1;1;1] when represented in a frame rotated by 90 degrees is at position [1;-1;1]
        result_z = np.matmul(rot90.RotZ3D(self, np.deg2rad(90)), initial_point)
        expected_result = np.array([[1], [-1], [1]])
        np.testing.assert_array_almost_equal(result_z, expected_result)

    def test_rotation_from_point_in_reference_frame_to_odom_frame(self):
        initial_point = np.array([[1], [1], [1]])
        rot90 = LA

        # Expected result with transpose Rotation in X
        result_x = np.matmul(rot90.RotX3D(self, np.deg2rad(90)).transpose(), initial_point)
        expected_result = np.array([[1], [-1], [1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with transpose Rotation in Y
        result_x = np.matmul(rot90.RotY3D(self, np.deg2rad(90)).transpose(), initial_point)
        expected_result = np.array([[1], [1], [-1]])
        np.testing.assert_array_almost_equal(result_x, expected_result)

        # Expected result with transpose Rotation in Z
        # The point [-1;1;1] when represented in a frame rotated by 90 degrees in Z is at position [1;1;1]
        result_z = np.matmul(rot90.RotZ3D(self, np.deg2rad(90)).transpose(), initial_point)
        expected_result = np.array([[-1], [1], [1]])
        np.testing.assert_array_almost_equal(result_z, expected_result)

    def test_transformation_matrix(self):
        initial_point = np.array([[1], [1], [1]])
        rot = LA

        result_transform = np.matmul(rot.TransformMatrix2D(self, np.deg2rad(0), initial_point), initial_point)
        np.testing.assert_array_equal(np.array([[2], [2], [1]]), result_transform)

        # Rotation and then Translation
        result_transform = np.matmul(rot.TransformMatrix2D(self, np.deg2rad(90), initial_point), initial_point)
        np.testing.assert_array_almost_equal(np.array([[2], [0], [1]]), result_transform)


unittest.main()