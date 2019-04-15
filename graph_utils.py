
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



def load_graph_file(data_file_fpath):
	"""
	"""
	scipy_obj = loadmat(data_file_fpath)
	x = scipy_obj['g'][0,0][0]
	edges = scipy_obj['g'][0,0][1]
	offset_dim_arr = scipy_obj['g'][0,0][2][0,0]
	idLookup_fieldnames = scipy_obj['idLookup_fieldnames']

	g = {}
	g['x'] = x
	g['edges'] = edges

	idLookup_fieldnames_dict = {}
	for i in range(len(idLookup_fieldnames)):
		fieldname = int(idLookup_fieldnames[i][0][0])
		idLookup_fieldnames_dict[fieldname] = i

	g['idLookup_fieldnames_dict'] = idLookup_fieldnames_dict
	g['offset_dim_arr'] = offset_dim_arr
	return g


def get_poses_landmarks(g):
	"""
	Extract the offset of the poses and the landmarks.

		Args:
		-	g

		Returns:
		-	poses
		-	landmarks
	"""
	poses = []
	landmarks = []

	for i in range(len(g['edges'])):
		from_idx =  g['edges'][i]['from'][0][0][0]
		offset_dim_arr_idx = g['idLookup_fieldnames_dict'][from_idx]
		value = g['offset_dim_arr'][offset_dim_arr_idx]
		dim = int(value['dimension'])
		offset = int(value['offset'])
		if (dim == 3):
			poses += [offset]
		elif (dim == 2):
			landmarks += [offset]
	return np.array(poses), np.array(landmarks)



def nnz_of_graph(g):
	"""
	Calculates an upper bound on the number of non-zeros of a graph,
	as duplicate edges might be counted several times

		Args:
		-	g

		Returns:
		-	nnz: integer, number of non-zeros of a graph
	"""
	nnz = 0

	# elements along the diagonal
	for i in range(len(g['edges'])):

		from_idx =  g['edges'][i]['from'][0][0][0]
		offset_dim_arr_idx = g['idLookup_fieldnames_dict'][from_idx]
		value = g['offset_dim_arr'][offset_dim_arr_idx]
		dim = int(value['dimension'])
		offset = int(value['offset'])
		nnz += dim**2

	# off-diagonal elements
	for eid in range(len(g['edges'])):
		edge = g['edges'][eid]
		edge_type = edge['type'][0][0]
		if edge_type=='P':
			# 3x3 block for Jacobian
			nnz += 2 * 9
		elif edge_type=='L':
			# 3x2 block for Jacobian
			nnz += 2 * 6

	return nnz



def plot_graph(fig, g, iteration=-1):
	"""
	plot a 2D SLAM graph
	"""
	p, l = get_poses_landmarks(g)

	plt.clf()

	if len(l) > 0:
		landmarkIdxX = l+1
		landmarkIdxY = l+2
		plt.scatter(g['x'][landmarkIdxX], g['x'][landmarkIdxY], 4, color='r', marker='o')

	if len(p) > 0:
		pIdxX = p
		pIdxY = p+1
		plt.scatter(g['x'][pIdxX], g['x'][pIdxY], 4, color='b', marker='x')
	fig.canvas.draw()
	plt.pause(1)
	# plt.show()
	# plt.pause(1)
	# plt.close('all')
	# # draw line segments???
	# if 0
	# 	poseEdgesP1 = [];
	# 	poseEdgesP2 = [];
	# 	landmarkEdgesP1 = [];
	# 	landmarkEdgesP2 = [];
	# 	for eid = 1:length(g.edges)
	# 		edge = g.edges(eid);
	# 		if (strcmp(edge.type, 'P') != 0):
	# 			poseEdgesP1 = [poseEdgesP1, g.x(edge.fromIdx:edge.fromIdx+1)]
	# 			poseEdgesP2 = [poseEdgesP2, g.x(edge.toIdx:edge.toIdx+1)]
	# 		elif (strcmp(edge.type, 'L') != 0):
	# 			landmarkEdgesP1 = [landmarkEdgesP1, g.x(edge.fromIdx:edge.fromIdx+1)]
	# 			landmarkEdgesP2 = [landmarkEdgesP2, g.x(edge.toIdx:edge.toIdx+1)]

	# 	linespointx = [poseEdgesP1(1,:); poseEdgesP2(1,:)]
	# 	linespointy = [poseEdgesP1(2,:); poseEdgesP2(2,:)]

	# 	plot(linespointx, linespointy, "r")


	#plot(poseEdgesP1(1,:), poseEdgesP1(2,:), "r");

	#if (columns(poseEdgesP1) > 0)
	#end
	#if (columns(landmarkEdges) > 0)
	#end


	# plt.figure() #, "visible", "on");
	#drawnow;
	#pause(0.1);
	if (iteration >= 0):
		filename = f'plots/lsslam_{iteration:03d}.png'
		#print(filename, '-dpng');
		plt.savefig(filename)









# def read_graph(filename):
# 	"""
# 	read a g2o data file describing a 2D SLAM instance

# 		Args:
# 		-	filename
# 		Returns:
# 		-	graph
# 	"""
# 	fid = fopen(filename, 'r');

# 	graph = struct (
# 	  'x', [],
# 	  'edges', [],
# 	  'idLookup', struct
# 	);

# 	disp('Parsing File');
# 	while true
# 	  ln = fgetl(fid);
# 	  if (ln == -1)
# 	    break;
# 	  end
# 	  tokens = strsplit(ln, ' ', true);
# 	  double_tokens = str2double(tokens);

# 	  tk = 2;
# 	  if (strcmp(tokens(1), 'VERTEX_SE2') != 0):
# 	    id = int32(double_tokens(tk++));
# 	    values = double_tokens(tk:tk+2)'; tk += 3;
# 	    graph.idLookup = setfield(graph.idLookup, num2str(id), struct('offset', length(graph.x), 'dimension', length(values)));
# 	    graph.x = [graph.x; values];
# 	  elseif (strcmp(tokens(1), 'VERTEX_XY') != 0):
# 	    id = int32(double_tokens(tk++));
# 	    values = double_tokens(tk:tk+1)'; tk += 2;
# 	    graph.idLookup = setfield(graph.idLookup, num2str(id), struct('offset', length(graph.x), 'dimension', length(values)));
# 	    graph.x = [graph.x; values];
# 	  elseif (strcmp(tokens(1), 'EDGE_SE2') != 0):
# 	    fromId = int32(double_tokens(tk++));
# 	    toId = int32(double_tokens(tk++));
# 	    measurement = double_tokens(tk:tk+2)'; tk += 3;
# 	    uppertri = double_tokens(tk:tk+5)'; tk += 6;
# 	    information = [uppertri(1), uppertri(2), uppertri(3);
# 	                   uppertri(2), uppertri(4), uppertri(5);
# 	                   uppertri(3), uppertri(5), uppertri(6)];
# 	    graph.edges = [graph.edges; struct(
# 	      'type', 'P',
# 	      'from', fromId,
# 	      'to', toId,
# 	      'measurement', measurement,
# 	      'information', information)];
# 	  elseif (strcmp(tokens(1), 'EDGE_SE2_XY') != 0):
# 	    fromId = int32(double_tokens(tk++));
# 	    toId = int32(double_tokens(tk++));
# 	    measurement = double_tokens(tk:tk+1)'; tk += 2;
# 	    uppertri = double_tokens(tk:tk+2)'; tk += 3;
# 	    information = [uppertri(1), uppertri(2); uppertri(2), uppertri(3)];
# 	    graph.edges = [graph.edges; struct(
# 	      'type', 'L',
# 	      'from', fromId,
# 	      'to', toId,
# 	      'measurement', measurement,
# 	      'information', information)];
# 	  end

# 	end

# 	% setup the index into the state vector
# 	disp('Preparing helper structs');
# 	for eid = 1:length(graph.edges):
# 	  graph.edges(eid).fromIdx = getfield(graph.idLookup, num2str(graph.edges(eid).from)).offset + 1;
# 	  graph.edges(eid).toIdx = getfield(graph.idLookup, num2str(graph.edges(eid).to)).offset + 1;


# 	return graph




