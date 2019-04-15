
import numpy as np
import pdb

from se2_utils import v2t, t2v, invt

def linearize_pose_pose_constraint(v_i, v_j, z_ij):
	"""
	% Compute the error of a pose-pose constraint
	% x1 3x1 vector (x,y,theta) of the first robot pose
	% x2 3x1 vector (x,y,theta) of the second robot pose
	% z 3x1 vector (x,y,theta) of the measurement
	%
	% You may use the functions v2t() and t2v() to compute
	% a Homogeneous matrix out of a (x, y, theta) vector
	% for computing the error.
	%  % extract the poses of the vertices and the mean of the edge
	    % computes the taylor expansion of the error function of the k_th edge

	Returns:
	-	e: 3x1 error of the constraint -  e_k(x)
	-	A: 3x3 Jacobian wrt x1 - d e_k(x) / d(x_i)
	-	B: 3x3 Jacobian wrt x2 -  d e_k(x) / d(x_j)
	"""
	# compute the homogeneous transforms of the previous solutions
	zt_ij = v2t(z_ij)
	vt_i = v2t(v_i)
	vt_j = v2t(v_j)
	
	# compute the displacement between x_i and x_j
	f_ij = np.linalg.inv(vt_i).dot(vt_j)
	
	# this below is too long to explain, to understand it derive it by hand
	theta_i = v_i[2,0]
	ti = v_i[:2]
	tj = v_j[:2]
	dt_ij = tj - ti

	si = np.sin(theta_i)
	ci = np.cos(theta_i)

	R_i = np.array([[ ci, -si ], 
					[si, ci ]])

	dRT_dtheta = np.array([	[-si, ci],
							[-ci, -si]])

	A = np.hstack([ -R_i.T, dRT_dtheta.dot(dt_ij) ])
	A = np.vstack([	A, 
					np.array([[0,0,-1]])	])

	B = np.array([	[ci, si, 0], 
					[-si, ci, 0],
					[0, 0, 1 ]])

	ztinv = np.linalg.inv(zt_ij)
	e = t2v(ztinv.dot(f_ij))
	ztinv[:2,2] = 0
	A = ztinv.dot(A)
	B = ztinv.dot(B)
	return e, A, B




def linearize_pose_landmark_constraint(x, l, z):
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
	t_i = x[:2]
	theta_i = x[2,0]

	si = np.sin(theta_i)
	ci = np.cos(theta_i)
	R_i = np.array([[ ci, -si ], 
					[si, ci ]])

	e = R_i.T.dot(l-t_i) - z

	dRT_dtheta = np.array([	[-si, ci],
							[-ci, -si]])
	A = np.hstack([ -R_i.T,  dRT_dtheta.dot(l - t_i) ])
	B = R_i.T

	return e, A, B








