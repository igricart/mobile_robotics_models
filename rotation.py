import numpy as np

class Rotation:
    def __init__(self, theta: float) -> None:
        self._theta = theta

    @property
    def RotZ2D(self) -> np.array:
        return np.array( [[np.cos(self._theta), np.sin(self._theta)], [-np.sin(self._theta), np.cos(self._theta)]] )
    
    @property
    def RotX3D(self) -> np.array:
        # Takes reference from input to output. When theta is the rotation in X from the local frame relative to the global frame
        # Example: xi_local_frame = RotX3D * xi_global_frame
        return np.array( [[1, 0, 0],
                         [0, np.cos(self._theta), np.sin(self._theta)],
                         [0, -np.sin(self._theta), np.cos(self._theta)]] )

    @property
    def RotY3D(self) -> np.array:
        # Takes reference from input to output. When theta is the rotation in Y from the local frame relative to the global frame
        # Example: xi_local_frame = RotX3D * xi_global_frame
        return np.array( [[np.cos(self._theta), 0, -np.sin(self._theta)],
                         [0, 1, 0],
                         [np.sin(self._theta), 0, np.cos(self._theta)]] )

    @property
    def RotZ3D(self) -> np.array:
        # Takes reference from input to output. When theta is the rotation in Z from the local frame relative to the global frame
        # Example: xi_local_frame = RotZ3D * xi_global_frame
        return np.array( [[np.cos(self._theta), np.sin(self._theta), 0],
                         [-np.sin(self._theta), np.cos(self._theta), 0],
                         [0, 0, 1]] )

