
import pdb
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import csc_matrix

from error_jacobians import linearize_pose_landmark_constraint, linearize_pose_pose_constraint
from graph_utils import nnz_of_graph


def linearize_and_solve(g):
	"""
	Performs one iteration of the Gauss-Newton algorithm.
	Each constraint is linearized and added to the Hessian

	# ---- CHOLMOD PYTHON ----
	from scikits.sparse.cholmod import cholesky
	factor = cholesky(A)
	x = factor(b)
	Factor.solve_A(b)
	# ------------------------

		Args:
		-	g: Python dictionary, containing graph information, with pose
				variables in a vector "x" of shape (K,1)

		Returns:
		-	dx: Numpy array (K,1) representing changed pose variables("delta x")
	"""
	nnz = nnz_of_graph(g)

	# allocate the sparse H and the vector b
	#H = spalloc(length(g.x), length(g.x), nnz);

	# x already accounts for 3 values per vertex
	H = np.zeros((len(g['x']), len(g['x'])))

	# H=zeros(size(vmeans,2)*3,size(vmeans,2)*3);
	# b=zeros(size(vmeans,2)*3,1);

	b = np.zeros((len(g['x']), 1))
	needToAddPrior = True

	# compute the addend term to H and b for each of our constraints
	print('\tLinearize and build system')
	for eid in range(len(g['edges'])):
		edge = g['edges'][eid]

		# convert 1-indexed to 0-indexed
		i = int(edge['fromIdx']) - 1
		j = int(edge['toIdx']) - 1
		edge_type = edge['type'][0][0]

		# retrieve edge measurement
		z = edge['measurement'][0]
		# retrieve edge information matrix
		omega = edge['information'][0]

		# pose-pose constraint
		if edge_type=='P':
			# edge.fromIdx and edge.toIdx describe the location of
			# the first element of the pose in the state vector

			# edge.information is the information matrix
			x1 = g['x'][i:i+3]  # the first robot pose
			x2 = g['x'][j:j+3]  # the second robot pose

			# Computing the error and the Jacobians
			# e the error vector. A Jacobian wrt x1. B Jacobian wrt x2
			[e, A, B] = linearize_pose_pose_constraint(x1, x2, z)

			# Update H matrix and vector b
			# compute the blocks of H^k
			b_i = -A.T.dot(omega).dot(e)
			b_j = -B.T.dot(omega).dot(e)
			H_ii = A.T.dot(omega).dot(A)
			H_ij = A.T.dot(omega).dot(B)
			H_jj = B.T.dot(omega).dot(B)

			# accumulate the blocks in H and b
			#     H((id_i-1)*3+1:id_i*3,(id_i-1)*3+1:id_i*3) = H((id_i-1)*3+1:id_i*3,(id_i-1)*3+1:id_i*3) + H_ii;
			#     H((id_j-1)*3+1:id_j*3,(id_j-1)*3+1:id_j*3) = H((id_j-1)*3+1:id_j*3,(id_j-1)*3+1:id_j*3) + H_jj;
			#     H((id_i-1)*3+1:id_i*3,(id_j-1)*3+1:id_j*3) = H((id_i-1)*3+1:id_i*3,(id_j-1)*3+1:id_j*3) + H_ij;
			#     H((id_j-1)*3+1:id_j*3,(id_i-1)*3+1:id_i*3)= H((id_j-1)*3+1:id_j*3,(id_i-1)*3+1:id_i*3) + H_ij';
			#     b((id_i-1)*3+1:id_i*3,1) = b((id_i-1)*3+1:id_i*3,1) + b_i;
			#     b((id_j-1)*3+1:id_j*3,1) = b((id_j-1)*3+1:id_j*3,1) + b_j;

			H[i:i+3,i:i+3] += H_ii;
			H[j:j+3,j:j+3] += H_jj;
			H[i:i+3,j:j+3] += H_ij;
			H[j:j+3,i:i+3] += H_ij.T

			b[i:i+3] += b_i
			b[j:j+3] += b_j

			#     b((id_i-1)*3+1:id_i*3,1) = b((id_i-1)*3+1:id_i*3,1) + b_i;
			#     b((id_j-1)*3+1:id_j*3,1) = b((id_j-1)*3+1:id_j*3,1) + b_j;

			if needToAddPrior:
				# TODO: add the prior for one pose of this edge
				# This fixes one node to remain at its current location
				needToAddPrior = False

		# pose-landmark constraint
		elif edge_type=='L':
			# edge.fromIdx and edge.toIdx describe the location of
			# the first element of the pose and the landmark in the state vector
			# You should use also this index when updating the elements
			# of the H matrix and the vector b.


			x1 = g['x'][i:i+3]  # the robot pose
			x2 = g['x'][j:j+2]  # the landmark

			# Computing the error and the Jacobians
			# e the error vector
			# A Jacobian wrt x1
			# B Jacobian wrt x2
			[e, A, B] = linearize_pose_landmark_constraint(x1, x2, z)

			# compute the blocks of H^k
			# Update H matrix and vector b
			# compute the blocks of H^k
			b_i = -A.T.dot(omega).dot(e)
			b_j = -B.T.dot(omega).dot(e)
			H_ii = A.T.dot(omega).dot(A)
			H_ij = A.T.dot(omega).dot(B)
			H_jj = B.T.dot(omega).dot(B)

			# w.r.t. i should be 3
			# w.r.t. j should be 2

			# accumulate the blocks in H and b
			H[i:i+3,i:i+3] += H_ii;
			H[j:j+2,j:j+2] += H_jj;
			H[i:i+3,j:j+2] += H_ij;
			H[j:j+2,i:i+3] += H_ij.T

			b[i:i+3] += b_i
			b[j:j+2] += b_j

	# We solve the linear system, solution stored in dx.
	# Instead of inverting H explicitly, we use the backslash operator

	# note that the system (H b) is obtained only from
	# relative constraints. H is not full rank.
	# we solve the problem by anchoring the position of
	# the the first vertex.
	# this can be expressed by adding the equation 
	#   deltax(1:3,1)=0;
	# which is equivalent to the following
	H[:3,:3] += np.eye(3)

	print('\tSystem size: ', H.shape )
	print('\tSolving (may take some time) ...');
	#SH=sparse(H)
	#dx=SH\b

	# Form Compressed Sparse Column (CSC) matrix
	SH = csc_matrix(H)
	dx = scipy.sparse.linalg.spsolve(SH,b)

	print('\tLinear Solve Done! ')
	return dx.reshape(-1,1)

