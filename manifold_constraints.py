"""

Author: John Lambert
"""

import numpy as np


def clip_angle_to_arctan(theta: float) -> float:
    """
    Move to [-pi,pi] range
    """
    s = np.sin(theta)
    c = np.cos(theta)
    return np.arctan2(s, c)


def normalize_angles(g):
    """
    Normalize the angles between -PI and PI
    Only use robot poses that do not correspond to landmark poses
            (those have a 3rd element theta)

    Args:
        g: graph, with poses OFF the manifold SO(2)

    Returns:
        g:graph, with poses ON the manifold SO(2)
    """
    for v_id, v_pgo in g.vertex_map.items():
        if v_pgo.dim == 3:

            # get the offset of the x element of this pose vector
            # in the whole state vector
            offs = v_pgo.x_offset_idx

            # theta from the first robot pose
            g.x[offs + 2] = clip_angle_to_arctan(g.x[offs + 2])

    return g
