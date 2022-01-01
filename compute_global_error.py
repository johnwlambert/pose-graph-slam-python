"""

Author: John Lambert
"""

import pdb
import numpy as np
from se2_utils import SE2


def pose_pose_constraint_error(x_i: np.ndarray, x_j: np.ndarray, z_ij: np.ndarray):
    """
    
    Args:
        x_i:
        x_j:
        z_ij:
    """
    pdb.set_trace()
    t_i = x_i[:2]
    t_j = x_j[:2]
    t_ij = z_ij[:2]

    theta_i = x_i[2]
    theta_j = x_j[2]
    theta_ij = z_ij[2]

    si = np.sin(theta_i)
    ci = np.cos(theta_i)
    R_i = np.array([[ci, -si], [si, ci]])

    sij = np.sin(theta_ij)
    cij = np.cos(theta_ij)
    R_ij = np.array([[cij, -sij], [sij, cij]])

    e_ij = np.array([R_ij.T.dot(R_i.T).dot(t_j - t_i) - t_ij, theta_j - theta_i - theta_ij])
    return e_ij


def pose_landmark_constraint_error(x_i: np.ndarray, x_l: np.ndarray, z_il: np.ndarray):
    """
    Args:
        x_i:
        x_l:
        z_il:
    """
    pdb.set_trace()
    t_i = x_i[:2]
    theta_i = x_i[2]

    si = np.sin(theta_i)
    ci = np.cos(theta_i)
    R_i = np.array([[ci, -si], [si, ci]])

    e_il = R_i.T.dot(x_l - t_i) - z_il
    return e_il


def compute_global_error(g):
    """
    Computes the total error of the graph.

    Args:
        g

    Returns:
        Fx
    """
    Fx = 0

    # Loop over all edges
    for eid in range(len(g["edges"])):
        edge = g["edges"][eid]
        edge_type = edge["type"][0][0]

        # convert from 1-indexed to 0-indexed
        i = int(edge["fromIdx"]) - 1
        j = int(edge["toIdx"]) - 1

        # pose-pose constraint
        if edge_type == "P":

            x_i = v2t(g["x"][i : i + 3].reshape(3, 1))  # the first robot pose
            x_j = v2t(g["x"][j : j + 3].reshape(3, 1))  # the second robot pose
            pdb.set_trace()
            # TODO compute the error of the constraint and add it to Fx.
            # Use edge.measurement and edge.information to access the
            # measurement and the information matrix respectively.

            e_ij = pose_pose_constraint_error(x_i, x_j, z_ij)

        # pose-landmark constraint
        elif edge_type == "L":
            x_i = g["x"][i : i + 3]  # the robot pose
            x_l = g["x"][j : j + 2]  # the landmark

            # TODO compute the error of the constraint and add it to Fx.
            # Use edge.measurement and edge.information to access the
            # measurement and the information matrix respectively.

            e_il = pose_landmark_constraint_error(x_i, x_l, z_il)

    return Fx
