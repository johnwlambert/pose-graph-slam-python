
"""

Author: John Lambert
"""

import time
import pdb

import numpy as np
import matplotlib.pyplot as plt

import gauss_newton
import manifold_constraints
from pose_graph import PoseGraph2D, plot_graph, plot_graph_connectivity
from compute_global_error import compute_global_error


def run_lsslam(solver, g, dataset_name: str, numIterations: int = 100):
	"""Run least squares SLAM.
	
	Args:
	    solver:
	    g: graph
	    dataset_name:
        numIterations: the number of iterations of Gauss-Newton
	"""
	visualize=True
	if visualize==True:
		fig = plt.gcf()
		fig.show()
		fig.canvas.draw()
		# plot the initial state of the graph
		plot_graph(fig, g, 0)

	#print('Initial error %f\n', compute_global_error(g))
	# maximum allowed dx
	EPSILON = 10**-4

	# Error
	err = 0

	# carry out the iterations
	for i in range(1,numIterations):
		print(f'Performing iteration {i}')
		start = time.time()
		dx = gauss_newton.linearize_and_solve(g, i, dataset_name, solver)
		end = time.time()
		duration = end - start
		print(f'Iter {i} took {duration:.3f} sec.')

		# Apply the solution to the state vector g.x
		g.x += dx

		g = manifold_constraints.normalize_angles(g)
		if visualize==True:
			# plot the current state of the graph
			plot_graph(fig, g, i)
		#err = compute_global_error(g)

		# Print current error
		print(f'Current error {err}')

		# TODO: implement termination criterion as suggested on the sheet
		if (np.linalg.norm(dx) < 1e-10):
			break

	print(f'Final error {err}')


if __name__ == '__main__':
	"""
	DLR is Deutsches Zentrum für Luft- und Raumfahrt (German Aerospace Center)
	"""
	data_dir = '/Users/jlambert-admin/Documents/GeorgiaTech/CS_6643/pose-graph-slam/data'
	
	# simulation datasets
	#dataset_name = 'simulation-pose-landmark'
	#dataset_name = 'simulation-pose-pose'

	# real-world datasets
	#dataset_name = 'intel'
	dataset_name = 'dlr'

	g = PoseGraph2D(dataset_name)

	# solver = 'cg'
	solver = 'sparse_scipy_solver'

	#plot_graph_connectivity(g)
	run_lsslam(solver, g, dataset_name)

