
import numpy as np
import pdb

def v2t(v):
	"""
	Computes the homogeneous transform matrix A of the pose vector v.
		Args:
		-	v: (3,1) vector
		Returns:
		-	A
	"""
	v = v.squeeze()
	c = np.cos(v[2])
	s = np.sin(v[2])
	A = np.array([	[c, -s, v[0] ],
					[s,  c, v[1] ],
					[0., 0., 1.  ]])
	return A


def t2v(A):
	"""
	Computes the pose vector v from a homogeneous transform A.
		Args:
		-	A
		Returns:
		-	v
	"""
	v = np.zeros(3)
	theta = np.arctan2(A[1,0],A[0,0])
	v[:2] = A[:2,2]
	v[2] = theta
	return v.reshape(3,1)

def invt(m):
	"""
	Inverts a homogenous transform.
		Args:
		-	m
		Returns:
		-	A
	"""
	R = m[:2,:2]
	t = m[:2,2]

	A = np.eye(3)
	A[:2,:2] = R.T
	A[:2,2] = -R.T.dot(t)
	return A
