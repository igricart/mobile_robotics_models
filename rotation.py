import numpy as np

class LA:
    # theta in radians
    def __init__(self) -> None:
        pass

    def RotZ2D(self, theta: float) -> np.array:
        return np.array( [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]] )
    
    def RotX3D(self, theta: float) -> np.array:
        # Takes reference from input to output. When theta is the rotation in X from the local frame relative to the global frame
        # Example: xi_local_frame = RotX3D * xi_global_frame
        return np.array( [[1, 0, 0],
                         [0, np.cos(theta), np.sin(theta)],
                         [0, -np.sin(theta), np.cos(theta)]] )

    def RotY3D(self, theta: float) -> np.array:
        # Takes reference from input to output. When theta is the rotation in Y from the local frame relative to the global frame
        # Example: xi_local_frame = RotX3D * xi_global_frame
        return np.array( [[np.cos(theta), 0, -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0, np.cos(theta)]] )

    def RotZ3D(self, theta: float) -> np.array:
        # Takes reference from input to output. When theta is the rotation in Z from the local frame relative to the global frame
        # Example: xi_local_frame = RotZ3D * xi_global_frame
        return np.array( [[np.cos(theta), np.sin(theta), 0],
                         [-np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]] )

    def TransformMatrix2D(self, theta: float, xy_init: np.array) -> np.array:
        return np.array( [[np.cos(theta), np.sin(theta), xy_init[0]],
                         [-np.sin(theta), np.cos(theta), xy_init[1]],
                         [0, 0, 1]] )

