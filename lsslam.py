
import pdb
import numpy as np
import matplotlib.pyplot as plt

from graph_utils import load_graph_file, plot_graph
from gauss_newton import linearize_and_solve
from compute_global_error import compute_global_error



def clip_angle_to_arctan(theta):
	"""
	Move to [-pi,pi] range
	"""
	s = np.sin(theta)
	c = np.cos(theta)
	return np.arctan2(s,c)


def normalize_angles(g):
	"""
	Normalize the angles between -PI and PI
	Only use robot poses that do not correspond to landmark poses
		(those have a 3rd element theta)

		Args:
		-	g

		Returns:
		-	g
	"""
	for eid in range(len(g['edges'])):
		edge = g['edges'][eid]

		# convert 1-indexed to 0-indexed
		i = int(edge['fromIdx']) - 1
		j = int(edge['toIdx']) - 1
		edge_type = edge['type'][0][0]
		if edge_type=='P':
			# edge.fromIdx and edge.toIdx describe the location of
			# the first element of the pose in the state vector

			# edge.information is the information matrix
			g['x'][i+2] = clip_angle_to_arctan(g['x'][i+2])  # theta from the first robot pose
			g['x'][j+2] = clip_angle_to_arctan(g['x'][j+2])  # theta from the second robot pose

	return g




def run_lsslam(g, numIterations = 100):
	"""
		Args:
		-	g: graph
		-	numIterations: the number of iterations of Gauss-Newton
	"""
	fig = plt.gcf()
	fig.show()
	fig.canvas.draw()

	# plot the initial state of the graph
	plot_graph(fig, g, 0)

	print('Initial error %f\n', compute_global_error(g))
	# maximum allowed dx
	EPSILON = 10**-4

	# Error
	err = 0

	# carry out the iterations
	for i in range(numIterations):
		print(f'Performing iteration {i}')
		dx = linearize_and_solve(g)

		# Apply the solution to the state vector g.x
		g['x'] = g['x'] + dx

		g = normalize_angles(g)

		# plot the current state of the graph
		plot_graph(fig, g, i)
		err = compute_global_error(g)

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
	# dataset_name = 'simulation-pose-landmark'
	# dataset_name = 'simulation-pose-pose'

	# real-world datasets
	#dataset_name = 'intel'
	dataset_name = 'dlr'

	data_file_fpath = f'{data_dir}/{dataset_name}.mat'
	g = load_graph_file(data_file_fpath)

	run_lsslam(g)



