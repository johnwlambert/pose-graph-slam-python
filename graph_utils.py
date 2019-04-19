
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import networkx as nx


def load_graph_file(data_file_fpath, dataset_name):
	"""
	"""
	scipy_obj = loadmat(data_file_fpath)
	x = scipy_obj['g'][0,0][0]
	edges = scipy_obj['g'][0,0][1]
	offset_dim_arr = scipy_obj['g'][0,0][2][0,0]
	idLookup_fieldnames = scipy_obj['idLookup_fieldnames']

	g = {}
	g['x'] = x.squeeze()
	g['edges'] = edges

	idLookup_fieldnames_dict = {}
	for i in range(len(idLookup_fieldnames)):
		fieldname = int(idLookup_fieldnames[i][0][0])
		idLookup_fieldnames_dict[fieldname] = i

	g['idLookup_fieldnames_dict'] = idLookup_fieldnames_dict
	g['offset_dim_arr'] = offset_dim_arr

	vertices = []
	# Verify that the fromIdx and toIdx in an edge, are the same 
	# as the offset in vertex
	for v_id,i in g['idLookup_fieldnames_dict'].items():
		value = g['offset_dim_arr'][i]
		dim = int(value['dimension'])
		x_offset_idx = int(value['offset'])
		vertices += [VertexPGO(v_id, x_offset_idx, dim)]

	edges = []
	for edge in g['edges']:
		edge_type = str(edge['type'][0][0])
		from_v_id = int(edge['from'])
		to_v_id = int(edge['to'])
		measurement = edge['measurement'][0].squeeze()
		information = edge['information'][0]

		fromIdx = int(edge['fromIdx'])
		toIdx = int(edge['toIdx'])

		for v_pgo in vertices:
			if v_pgo.v_id == from_v_id:
				assert(fromIdx-1 == v_pgo.x_offset_idx)
			
			elif v_pgo.v_id == to_v_id:
				assert(toIdx-1 == v_pgo.x_offset_idx)

		edges += [EdgePGO(edge_type, from_v_id, to_v_id, measurement, information)]#, fromIdx, toIdx)]

	state_vec = g['x']
	pg2d = PoseGraph2D(edges, vertices, state_vec)
	write_graph_to_disk(dataset_name, pg2d)


	pdb.set_trace()
	
	return g



def write_graph_to_disk(dataset_name, g):
	"""
	"""
	edges_fpath = f'{dataset_name}_edges.txt'
	vertices_fpath = f'{dataset_name}_vertices.txt'
	initial_state_fpath = f'{dataset_name}_initial_state.txt'

	with open(edges_fpath, 'w') as f:
		for edge in g.edges:

			f.write(f'{edge.edge_type},')
			f.write(f'{edge.from_v_id},')
			f.write(f'{edge.to_v_id},')
			for z_i in edge.measurement:
				f.write(f'{z_i},')

			for ij, omega_ij in enumerate(edge.information.flatten()):
				f.write(f'{omega_ij}')
				if ij != len(edge.information.flatten())-1:
					f.write(',')
			f.write('\n')


	with open(vertices_fpath, 'w') as f:
		for v in g.vertices:
			f.write(f'{v.v_id},')
			f.write(f'{v.x_offset_idx},')
			f.write(f'{v.dim}\n')


	with open(initial_state_fpath, 'w') as f:
		for x_i in g.x:
			f.write(f'{x_i}\n')




def read_graph_from_disk(dataset_name):
	"""
	"""
	edges_fpath = f'{dataset_name}_edges.txt'
	vertices_fpath = f'{dataset_name}_vertices.txt'
	initial_state_fpath = f'{dataset_name}_initial_state.txt'

	with open(edges_fpath, 'r') as f:
		edges_data = f.readlines()

	edges = []
	for edge_info in edges_data:
		# print(edge_info)
		edge_info = edge_info.strip().split(',')
		edge_type = edge_info[0]
		from_v_id = int(edge_info[1])
		to_v_id = int(edge_info[2])

		if edge_type == 'L':
			dim = 2

		elif edge_type == 'P':
			dim = 3

		measurement = np.zeros(dim)
		for i,j in enumerate(range(3,3+dim)):
			measurement[i] = float(edge_info[j])
		information = np.zeros(dim*dim)
		for i,j in enumerate(range(3+dim,3+dim+dim**2)):
			information[i] = float(edge_info[j])
		information = information.reshape(dim,dim)
		edges += [EdgePGO(edge_type, from_v_id, to_v_id, measurement, information)]#, fromIdx, toIdx)]


	with open(vertices_fpath, 'r') as f:
		vertices_data = f.readlines()

	vertex_map = {}
	for line in vertices_data:
		line = line.strip().split(',')
		v_id = int(line[0])
		x_offset_idx = int(line[1])
		dim = int(line[2])
		vertex_map[v_id] = VertexPGO(v_id, x_offset_idx, dim)

	with open(initial_state_fpath, 'r') as f:
		state_data = f.readlines()
	x = np.zeros(len(state_data))

	for i in range(len(state_data)):
		x[i] = float(state_data[i])

	pg2d = PoseGraph2D(edges, vertex_map, x)
	return pg2d





class PoseGraph2D(object):
	""" """
	def __init__(self, edges, vertex_map, x):
		""" """
		self.edges = edges
		self.vertex_map = vertex_map

		# state vector (concatenated pose vectors)
		self.x = x


class VertexPGO(object):
	""" Pose Graph Vertex """
	def __init__(self, v_id, x_offset_idx, dim):
		""" 
		v_id is "vertex ID"
		"""
		self.v_id = v_id
		self.x_offset_idx = x_offset_idx
		self.dim = dim


class EdgePGO(object):
	""" Pose Graph Edge """
	def __init__(self, edge_type, from_v_id, to_v_id, measurement, information):#, fromIdx, toIdx):
		""" 
		"from_v_id" is "from" vertex ID
		"to_v_id" is "to" vertex ID
		"""
		self.edge_type = edge_type
		self.from_v_id = from_v_id
		self.to_v_id = to_v_id
		self.measurement = measurement
		self.information = information
		# self.fromIdx = fromIdx
		# self.toIdx = toIdx



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

	for v_id, v_pgo in g.vertex_map.items():
		if v_pgo.dim == 3:
			poses += [v_pgo.x_offset_idx]

		elif v_pgo.dim == 2:
			landmarks += [v_pgo.x_offset_idx]

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
	for (key, offset_dim_arr_idx) in g['idLookup_fieldnames_dict'].items():
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
	# retrieve indices into g's "x" vector -> start of each individual variable
	# plot (x,y) positions from each pose/landmark
	p, l = get_poses_landmarks(g)

	plt.clf()

	if len(l) > 0:
		landmarkIdxX = l
		landmarkIdxY = l+1
		plt.scatter(g.x[landmarkIdxX], g.x[landmarkIdxY], 4, color='r', marker='o')

	if len(p) > 0:
		pIdxX = p
		pIdxY = p+1
		plt.scatter(g.x[pIdxX], g.x[pIdxY], 4, color='b', marker='x')

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

	filename = f'plots/lsslam_{iteration:03d}.png'
	#print(filename, '-dpng');
	plt.savefig(filename)



def plot_graph_connectivity(g):
	"""
	"""
	G = nx.DiGraph()
	all_edges = []
	L_edges = []

	for eid in range(len(g['edges'])):
		edge = g['edges'][eid]

		# convert 1-indexed to 0-indexed
		i = int(edge['fromIdx']) - 1
		j = int(edge['toIdx']) - 1

		all_edges += [(i,j)]
		edge_type = edge['type'][0][0]
		if edge_type == 'L':
			L_edges += [(i,j)]

	G.add_edges_from(all_edges)

	#values = [val_map.get(node, 0.25) for node in G.nodes()]

	# Specify the edges you want here
	# edge_colours = ['black' if not edge in L_edges else 'red' for edge in G.edges()]
	P_edges = [edge for edge in G.edges() if edge not in L_edges]

	# Need to create a layout when doing
	# separate calls to draw nodes and edges
	#pos = nx.spring_layout(G)

	pos = nx.kamada_kawai_layout(G)


	nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 50)
	nx.draw_networkx_labels(G, pos)
	nx.draw_networkx_edges(G, pos, edgelist=P_edges, edge_color='k', arrows=False)
	nx.draw_networkx_edges(G, pos, edgelist=L_edges, edge_color='r', arrows=True)
	plt.show()





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




