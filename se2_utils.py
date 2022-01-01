"""

Author: John Lambert
"""

import numpy as np
import pdb


class SE2(object):
    def __init__(self, v: np.ndarray) -> None:
        """
        Computes the homogeneous transform matrix A of the pose vector v.

        Args:
            v: (3,1) vector

        Returns:
            A: 3x3 Numpy array

        Args:
            R: Numpy array of shape (2,2)
            t: Numpy array of shape (2,1)
        """
        assert v.shape == (3,)

        v = v.squeeze()
        self.v = v
        self.t = v[:2]
        self.theta = v[2]

        c = np.cos(self.theta)
        s = np.sin(self.theta)
        self.R = np.array([[c, -s], [s, c]])

        self.dRT_dtheta = np.array([[-s, c], [-c, -s]])
        # assert(R.shape == (2,2))
        # assert(t.shape == (2,))

        self.mat_3x3 = np.eye(3)
        self.mat_3x3[:2, :2] = self.R
        self.mat_3x3[:2, 2] = self.t

    def inverse(self) -> np.ndarray:
        """Inverts a homogenous transform.

        Returns:
            T_inv: 3x3 Numpy array
        """
        T_inv = np.eye(3)
        T_inv[:2, :2] = self.R.T
        T_inv[:2, 2] = -self.R.T.dot(self.t)
        return T_inv


class SE2_mat(object):
    def __init__(self, mat_3x3: np.ndarray) -> None:
        self.R = mat_3x3[:2, :2]
        self.t = mat_3x3[:2, 2]

    def as_pose_vector(self):
        """
        Computes the pose vector v from a homogeneous transform A.

        Args:
            A: 3x3 Numpy array

        Returns:
            v
        """
        v = np.zeros(3)
        v[:2] = self.t
        theta = np.arctan2(self.R[1, 0], self.R[0, 0])
        v[2] = theta
        return v  # .reshape(3,1)


# class PoseVector(object):
# 	def __init__(self, v):


# 	def as_SE2(self):
# 		"""
