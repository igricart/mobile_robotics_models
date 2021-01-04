from rotation import Rotation
import numpy as np
class MobileRobot:
    def __init__(self, state: np.array, dt: float, wheel_radius: float = 1.0, distance_between_wheel_and_rotation_point: float = 1.0) -> None:
        if state.size != 3:
            print("Current number of states: ", state.size,". Mobile Robot should have 3 states...")
            exit()
        else:
            self._dt = dt
            self._x = state[0]
            self._y = state[1]
            # reference in odom
            self._theta = state[2]
            self._r = wheel_radius
            self._l = distance_between_wheel_and_rotation_point # distance from robot center of rotation and wheel
            self._cumulative_dt = 0
        
    def update_local_frame(self, input: np.array) -> np.array:
        # input is velocity on each wheel
        if input.size == 2:
            omega_right = input[0] * self._r / (2 * self._l)
            omega_left = - input[1] * self._r / (2 * self._l)

            return np.array([[self._r * (input[0] + input[1]) / 2],
                                [0], 
                                [omega_right + omega_left]])
        else:
            print("There should be a np.array of size 2 as input...")
            exit()

    def update(self, input: np.array) -> np.array:
        local_frame_states = self.update_local_frame(input)
        rot = Rotation(self._theta)
        odom_frame_states = np.matmul(rot.RotZ3D.transpose(), local_frame_states)
        
        #print("My local frame states are:\n", local_frame_states)
        #print("My odom frame states are:\n",odom_frame_states)
        
        return odom_frame_states

    def update_method_integral(self, input: np.array) -> np.array:
        # calculating in relation to initial position and formula is considering the whole time
        self._cumulative_dt += self._dt
        self._theta = (input[0] - input[1]) * self._cumulative_dt / (self._l * 2)
        self._x = ((input[0] + input[1]) / (input[0] - input[1])) * self._l * np.sin(self._theta)
        self._y = ((input[0] + input[1]) / (input[0] - input[1])) * self._l * (np.cos(self._theta) + 1)
        return np.array([self._x, self._y, self._theta, self._cumulative_dt, self._dt])

    @property
    def states(self) -> np.array:
        return np.array([self._x, self._y, self._theta, self._dt, self._r, self._l,])