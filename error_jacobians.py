
import numpy as np
import pdb

from se2_utils import SE2, SE2_mat


def linearize_pose_pose_constraint(v_i, v_j, z_ij):
	"""
	Compute the error of a pose-pose constraint. We 
	compute homogeneous matrix transforms from the pose
	vectors (x,y,theta).

	We extract the poses of the vertices and the mean of this k'th edge.
	Computes the taylor expansion of the error function of this k'th edge.

		Args:
		-	v_i: 3x1 vector (x,y,theta) of the first robot pose
		-	v_j: 3x1 vector (x,y,theta) of the second robot pose
		-	z 3x1 vector (x,y,theta) of the measurement

		Returns:
		-	e: 3x1 error of the constraint -  e_k(x)
		-	A: 3x3 Jacobian wrt x1 - d e_k(x) / d(x_i)
		-	B: 3x3 Jacobian wrt x2 -  d e_k(x) / d(x_j)
	"""

	# compute the homogeneous transforms of the previous solutions
	Z_ij  = SE2(z_ij)
	w_T_i = SE2(v_i)
	w_T_j = SE2(v_j)
	
	# compute the displacement between x_i and x_j
	f_ij = w_T_i.inverse().dot(w_T_j.mat_3x3)
	
	# this below is too long to explain, to understand it derive it by hand
	dt_ij = w_T_j.t - w_T_i.t

	A = -np.eye(3)
	A[:2,:2] = -w_T_i.R.T
	A[:2,2] = w_T_i.dRT_dtheta.dot(dt_ij)

	B = np.eye(3)
	B[:2,:2] = w_T_i.R.T
	B[:2,2] = np.zeros(2)

	ztinv = Z_ij.inverse()
	e = SE2_mat(ztinv.dot(f_ij)).as_pose_vector()
	ztinv[:2,2] = 0
	A = ztinv.dot(A)
	B = ztinv.dot(B)
	return e, A, B




def linearize_pose_landmark_constraint(x_i, l, z):
	"""
	Compute the error of a pose-landmark constraint
	and the Jacobians of the error.

	Args:
	-	x: 3x1 vector (x,y,theta) of the robot pose
	-	l: 2x1 vector (x,y) of the landmark
	-	z: 2x1 vector (x,y) of the measurement, the position of the landmark in
			the coordinate frame of the robot given by the vector x

	Output
	-	e: 2x1 error of the constraint
	-	A: 2x3 Jacobian wrt x
	-	B: 2x2 Jacobian wrt l
	"""
	l = l.squeeze()
	z = z.squeeze()

	w_T_i = SE2(x_i)
	e = w_T_i.R.T.dot(l - w_T_i.t) - z

	A = np.zeros((2,3))
	A[:2,:2] = -w_T_i.R.T
	A[:2,2] = w_T_i.dRT_dtheta.dot(l - w_T_i.t)
	B = w_T_i.R.T

	return e, A, B





