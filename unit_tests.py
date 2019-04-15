
import numpy as np
import pdb

from error_jacobians import linearize_pose_landmark_constraint, linearize_pose_pose_constraint
from se2_utils import v2t, t2v, invt

def test_jacobian_pose_landmark():
	""" """
	epsilon = 1e-5

	x1 = np.array([1., 1., 1.]).reshape(3,1)
	x2 = np.array([2.2, 1.9]).reshape(2,1)
	z  = np.array([1.3, -0.4]).reshape(2,1)

	# get the analytic Jacobian
	e, A, B = linearize_pose_landmark_constraint(x1, x2, z)

	# check the error vector
	e_true = np.array([0.10569, -0.12349]).reshape(2,1)
	if (np.linalg.norm(e - e_true) > epsilon):
		print('Your error function seems to return a wrong value')
		print('Result of your function'); print(e)
		print('True value'); print(e_true)
	else:
		print('The computation of the error vector appears to be correct')

	# compute it numerically
	delta = 1e-6;
	scalar = 1 / (2*delta);

	# test for x1
	ANumeric = np.zeros((2,3))
	for d in range(3):
		curX = x1.copy()
		curX[d] += delta
		err,_,_ = linearize_pose_landmark_constraint(curX, x2, z)
		curX = x1
		curX[d] -= delta
		err -= linearize_pose_landmark_constraint(curX, x2, z)[0]

		ANumeric[:, d] = scalar * err.squeeze()

	diff = ANumeric - A
	if np.amax(np.absolute(diff)) > epsilon:
		print('Error in the Jacobian for x1')
		print('Your analytic Jacobian'); print(A)
		print('Numerically computed Jacobian'); print(ANumeric)
		print('Difference'); print(diff)
	else:
		print('Jacobian for x1 appears to be correct');

	# test for x2
	BNumeric = np.zeros((2,2))
	for d in range(2):
		curX = x2.copy()
		curX[d] += delta
		err,_,_ = linearize_pose_landmark_constraint(x1, curX, z)
		curX = x2;
		curX[d] -= delta;
		err = err - linearize_pose_landmark_constraint(x1, curX, z)[0]
		BNumeric[:, d] = scalar * err.squeeze()
	
	diff = BNumeric - B
	if np.amax(abs(diff)) > epsilon:
		print('Error in the Jacobian for x2')
		print('Your analytic Jacobian'); print(B)
		print('Numerically computed Jacobian'); print(BNumeric)
		print('Difference'); print(diff);
	else:
		print('Jacobian for x2 appears to be correct')



def test_jacobian_pose_pose():
	""" """
	epsilon = 1e-5

	x1 = np.array([1., 1., 1.]).reshape(3,1)
	x2 = np.array([2.2, 1.85, 1.2]).reshape(3,1)
	z  = np.array([0.9, 1.1, 1.05]).reshape(3,1)

	pdb.set_trace()
	# get the analytic Jacobian
	[e, A, B] = linearize_pose_pose_constraint(x1, x2, z)

	# check the error vector
	e_true = np.array([-1.20101, -1.22339, -0.85000]).reshape(3,1)
	if (np.linalg.norm(e - e_true) > epsilon):
		print('Your error function seems to return a wrong value')
		print('Result of your function'); print(e)
		print('True value'); print(e_true)
	else:
		print('The computation of the error vector appears to be correct')

	# compute it numerically
	delta = 1e-6;
	scalar = 1 / (2*delta)

	# test for x1
	ANumeric = np.zeros((3,3))
	for d in range(3):
		curX = x1.copy()
		curX[d] += delta
		err,_,_ = linearize_pose_pose_constraint(curX, x2, z)
		curX = x1
		curX[d] -= delta
		err = err - linearize_pose_pose_constraint(curX, x2, z)[0]

		ANumeric[:, d] = scalar * err.squeeze()

	diff = ANumeric - A
	if np.amax(abs(diff)) > epsilon:
	  print('Error in the Jacobian for x1')
	  print('Your analytic Jacobian'); print(A)
	  print('Numerically computed Jacobian'); print(ANumeric);
	  print('Difference'); print(diff)
	else:
	  print('Jacobian for x1 appears to be correct')

	# test for x2
	BNumeric = np.zeros((3,3))
	for d in range(3):
		curX = x2.copy()
		curX[d] += delta
		err,_,_ = linearize_pose_pose_constraint(x1, curX, z)
		curX = x2
		curX[d] -= delta
		err = err - linearize_pose_pose_constraint(x1, curX, z)[0]

		BNumeric[:, d] = scalar * err.squeeze()

	diff = BNumeric - B
	if np.amax(abs(diff)) > epsilon:
		print('Error in the Jacobian for x2')
		print('Your analytic Jacobian'); print(B)
		print('Numerically computed Jacobian'); print(BNumeric)
		print('Difference'); print(diff)
	else:
		print('Jacobian for x2 appears to be correct')


if __name__ == '__main__':
	test_jacobian_pose_pose()
	test_jacobian_pose_landmark()
	





