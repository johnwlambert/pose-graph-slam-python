
"""

Author: John Lambert
"""

import time
import cv2
import pdb
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import csc_matrix

from error_jacobians import linearize_pose_landmark_constraint, linearize_pose_pose_constraint
from pose_graph import nnz_of_graph

# from iterative_solvers import cg


def linearize_and_solve(g, iter, dataset_name, solver):
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
    # allocate the sparse H and the vector b
    # H = spalloc(length(g.x), length(g.x), nnz);

    # x already accounts for 3 values per vertex
    H = np.zeros((g.x.size, g.x.size))

    analyze_nnz = False
    if analyze_nnz:
        nnz = nnz_of_graph(g)
        print("H has shape ", H.shape, " #elements in H is ", H.shape[0] * H.shape[1], " with ", nnz, " nnz")
        nnz_entry_percent = nnz / (H.shape[0] * H.shape[1]) * 100
        print("NNZ % = ", nnz_entry_percent)
        print("Not NNZ % = ", 100.0 - nnz_entry_percent)

    b = np.zeros(g.x.size)
    needToAddPrior = True

    # compute the addend term to H and b for each of our constraints
    print("\tLinearize and build system")
    for eid in range(len(g.edges)):

        edge = g.edges[eid]

        # convert 1-indexed to 0-indexed
        i_v_id = edge.from_v_id
        j_v_id = edge.to_v_id

        i = g.vertex_map[i_v_id].x_offset_idx
        j = g.vertex_map[j_v_id].x_offset_idx
        edge_type = edge.edge_type

        # retrieve edge measurement
        z = edge.measurement
        # retrieve edge information matrix
        omega = edge.information

        # pose-pose constraint
        if edge_type == "P":
            # edge.fromIdx and edge.toIdx describe the location of
            # the first element of the pose in the state vector

            # edge.information is the information matrix
            x1 = g.x[i : i + 3]  # the first robot pose
            x2 = g.x[j : j + 3]  # the second robot pose

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
            H[i : i + 3, i : i + 3] += H_ii
            H[j : j + 3, j : j + 3] += H_jj
            H[i : i + 3, j : j + 3] += H_ij
            H[j : j + 3, i : i + 3] += H_ij.T

            b[i : i + 3] += b_i
            b[j : j + 3] += b_j

            if needToAddPrior:
                # TODO: add the prior for one pose of this edge
                # This fixes one node to remain at its current location
                needToAddPrior = False

        # pose-landmark constraint
        elif edge_type == "L":
            # edge.fromIdx and edge.toIdx describe the location of
            # the first element of the pose and the landmark in the state vector
            # You should use also this index when updating the elements
            # of the H matrix and the vector b.

            x1 = g.x[i : i + 3]  # the robot pose
            x2 = g.x[j : j + 2]  # the landmark

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
            H[i : i + 3, i : i + 3] += H_ii
            H[j : j + 2, j : j + 2] += H_jj
            H[i : i + 3, j : j + 2] += H_ij
            H[j : j + 2, i : i + 3] += H_ij.T

            b[i : i + 3] += b_i
            b[j : j + 2] += b_j

    # We solve the linear system, solution stored in dx.
    # Instead of inverting H explicitly, we use the backslash operator

    # note that the system (H b) is obtained only from
    # relative constraints. H is not full rank.
    # we solve the problem by anchoring the position of
    # the the first vertex.
    # this can be expressed by adding the equation
    #   deltax(1:3,1)=0;
    # which is equivalent to the following
    H[:3, :3] += np.eye(3)

    # save_matrix_image( H.copy(), iter, dataset_name )

    print("\tSystem size: ", H.shape)
    print("\tSolving (may take some time) ...")
    # SH=sparse(H)
    # dx=SH\b

    b = b.reshape(-1, 1)
    # if solver == 'cg':
    # 	dx0 = np.zeros((len(g.x),1))
    # 	dx, _ = cg(H,b, dx0)

    if solver == "sparse_scipy_solver":
        # Form Compressed Sparse Column (CSC) matrix
        SH = csc_matrix(H)
        start = time.time()
        dx = scipy.sparse.linalg.spsolve(SH, b)
        duration = time.time() - start
        print(f"Linear solve took {duration:.2f}")

    print("\tLinear Solve Done! ")
    return dx.squeeze()


def save_matrix_image(H, iter, dataset_name):
    """ """
    cv2.imwrite(f"{dataset_name}_{iter}_H_matrix_original.png", H)
    n, _ = H.shape
    H = H.reshape(-1, 1)
    zero_bool = H == 0.0
    nnz_bool = np.logical_not(zero_bool)
    H[zero_bool] = 255
    H[nnz_bool] = 0.0
    H = H.reshape(n, n)
    cv2.imwrite(f"{dataset_name}_{iter}_H_matrix_nnz.png", H)
