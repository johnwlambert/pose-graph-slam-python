

import pdb
import numpy as np
from se2_utils import v2t, t2v, invt

def compute_global_error(g):
	"""
	Computes the total error of the graph.

		Args:
		-	g

		Returns:
		-	Fx
	"""
	Fx = 0

	# Loop over all edges
	for eid in range(len(g['edges'])):
		edge = g['edges'][eid]
		edge_type = edge['type'][0][0]

		# convert from 1-indexed to 0-indexed
		i = int(edge['fromIdx'])-1
		j = int(edge['toIdx'])-1

		# pose-pose constraint
		if edge_type == 'P':

			x1 = v2t(g['x'][i:i+3].reshape(3,1)) # the first robot pose
			x2 = v2t(g['x'][j:j+3].reshape(3,1)) # the second robot pose

			# TODO compute the error of the constraint and add it to Fx.
			# Use edge.measurement and edge.information to access the
			# measurement and the information matrix respectively.

		# pose-landmark constraint
		elif edge_type == 'L':
			pdb.set_trace()
			x = g['x'][i:i+3]  # the robot pose
			l = g['x'][j:j+2]  # the landmark

			#TODO compute the error of the constraint and add it to Fx.
			# Use edge.measurement and edge.information to access the
			# measurement and the information matrix respectively.

	return Fx
