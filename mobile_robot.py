import rotation
import numpy as np
class MobileRobot:
    def __init__(self, state: np.array, dt: float):
        if state.size != 3:
            print("Current number of states: ", state.size,". Mobile Robot should have 3 states...")
            exit()
        else:
            self._dt = dt
            self._x = state[0]
            self._y = state[1]
            # reference in odom
            self._theta = state[2]
        
    def update(self, input: np.array):
        # considering it is in the same reference frame
        # input is linear velocity and theta
        
        self._theta += input[2]

    @property
    def states(self) -> np.array:
        return np.array([self._x, self._y, self._theta])