"""

Author: John Lambert
"""

import os
import pdb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.io import loadmat


DATAROOT = Path(__file__).resolve().parent / "datasets"


class VertexPGO(object):
    """Pose Graph Vertex"""

    def __init__(self, v_id, x_offset_idx, dim):
        """
        v_id is "vertex ID"
        """
        self.v_id = v_id
        self.x_offset_idx = x_offset_idx
        self.dim = dim


class EdgePGO(object):
    """Pose Graph Edge"""

    def __init__(self, edge_type, from_v_id, to_v_id, measurement, information):  # , fromIdx, toIdx):
        """
        "from_v_id" is "from" vertex ID
        "to_v_id" is "to" vertex ID
        """
        self.edge_type = edge_type
        self.from_v_id = from_v_id
        self.to_v_id = to_v_id
        self.measurement = measurement
        self.information = information


class PoseGraph2D(object):
    """ """

    def __init__(self, dataset_name: str):
        """ """
        edges, vertex_map, x = self.read_graph_from_disk(dataset_name)

        self.edges = edges
        self.vertex_map = vertex_map

        # state vector (concatenated pose vectors)
        self.x = x

    def read_graph_from_disk(self, dataset_name: str):
        """ """
        edges_fpath = f"{DATAROOT}/{dataset_name}/{dataset_name}_edges.txt"
        vertices_fpath = f"{DATAROOT}/{dataset_name}/{dataset_name}_vertices.txt"
        initial_state_fpath = f"{DATAROOT}/{dataset_name}/{dataset_name}_initial_state.txt"

        edges = self.read_edge_data(edges_fpath)
        vertex_map = self.read_vertex_data(vertices_fpath)

        with open(initial_state_fpath, "r") as f:
            state_data = f.readlines()
        x = np.zeros(len(state_data))

        for i in range(len(state_data)):
            x[i] = float(state_data[i])

        return edges, vertex_map, x

    def read_edge_data(self, edges_fpath: str):
        """
        Args:
            edges_fpath

        Returns:
            edges: list of EdgePGO objects
        """
        with open(edges_fpath, "r") as f:
            edges_data = f.readlines()

        edges = []
        for edge_info in edges_data:
            # print(edge_info)
            edge_info = edge_info.strip().split(",")
            edge_type = edge_info[0]
            from_v_id = int(edge_info[1])
            to_v_id = int(edge_info[2])

            if edge_type == "L":
                dim = 2

            elif edge_type == "P":
                dim = 3

            measurement = np.zeros(dim)
            for i, j in enumerate(range(3, 3 + dim)):
                measurement[i] = float(edge_info[j])
            information = np.zeros(dim * dim)
            for i, j in enumerate(range(3 + dim, 3 + dim + dim ** 2)):
                information[i] = float(edge_info[j])
            information = information.reshape(dim, dim)
            edges += [EdgePGO(edge_type, from_v_id, to_v_id, measurement, information)]
        return edges

    def read_vertex_data(self, vertices_fpath: str):
        """

        Args:
            vertices_fpath:

        Returns:
            vertex_map: Python dictionary mapping vertex ID to VertexPGO objects
        """
        with open(vertices_fpath, "r") as f:
            vertices_data = f.readlines()

        vertex_map = {}
        for line in vertices_data:
            line = line.strip().split(",")
            v_id = int(line[0])
            x_offset_idx = int(line[1])
            dim = int(line[2])
            vertex_map[v_id] = VertexPGO(v_id, x_offset_idx, dim)
        return vertex_map

    def get_poses_landmarks(self):
        """
        Extract the offset of the poses and the landmarks.

        Args:
            g

        Returns:
            poses
            landmarks
        """
        poses = []
        landmarks = []
        for v_id, v_pgo in self.vertex_map.items():
            if v_pgo.dim == 3:
                poses += [v_pgo.x_offset_idx]
            elif v_pgo.dim == 2:
                landmarks += [v_pgo.x_offset_idx]

        return np.array(poses), np.array(landmarks)


def nnz_of_graph(g) -> int:
    """
    Calculates an upper bound on the number of non-zeros of a graph,
    as duplicate edges might be counted several times

    Args:
        g

    Returns:
        nnz: integer, number of non-zeros of a graph
    """
    nnz = 0

    # elements along the diagonal
    for (key, offset_dim_arr_idx) in g["idLookup_fieldnames_dict"].items():
        value = g["offset_dim_arr"][offset_dim_arr_idx]
        dim = int(value["dimension"])
        offset = int(value["offset"])
        nnz += dim ** 2

    # off-diagonal elements
    for eid in range(len(g["edges"])):
        edge = g["edges"][eid]
        edge_type = edge["type"][0][0]
        if edge_type == "P":
            # 3x3 block for Jacobian
            nnz += 2 * 9
        elif edge_type == "L":
            # 3x2 block for Jacobian
            nnz += 2 * 6

    return nnz


def plot_graph(fig, g, iteration: int = -1) -> None:
    """
    plot a 2D SLAM graph
    """
    # retrieve indices into g's "x" vector -> start of each individual variable
    # plot (x,y) positions from each pose/landmark
    p, l = g.get_poses_landmarks()

    plt.clf()

    if len(l) > 0:
        landmarkIdxX = l
        landmarkIdxY = l + 1
        plt.scatter(g.x[landmarkIdxX], g.x[landmarkIdxY], 4, color="r", marker="o")

    if len(p) > 0:
        pIdxX = p
        pIdxY = p + 1
        plt.scatter(g.x[pIdxX], g.x[pIdxY], 4, color="b", marker="x")

    fig.canvas.draw()
    plt.pause(1)

    os.makedirs("plots", exist_ok=True)
    filename = os.path.join("plots", f"lsslam_{iteration:03d}.png")
    plt.savefig(filename)


def write_graph_to_disk(dataset_name: str, g) -> None:
    """ """
    edges_fpath = f"{dataset_name}_edges.txt"
    vertices_fpath = f"{dataset_name}_vertices.txt"
    initial_state_fpath = f"{dataset_name}_initial_state.txt"

    with open(edges_fpath, "w") as f:
        for edge in g.edges:

            f.write(f"{edge.edge_type},")
            f.write(f"{edge.from_v_id},")
            f.write(f"{edge.to_v_id},")
            for z_i in edge.measurement:
                f.write(f"{z_i},")

            for ij, omega_ij in enumerate(edge.information.flatten()):
                f.write(f"{omega_ij}")
                if ij != len(edge.information.flatten()) - 1:
                    f.write(",")
            f.write("\n")

    with open(vertices_fpath, "w") as f:
        for v_id, v in self.vertex_map.items():
            f.write(f"{v.v_id},")
            f.write(f"{v.x_offset_idx},")
            f.write(f"{v.dim}\n")

    with open(initial_state_fpath, "w") as f:
        for x_i in g.x:
            f.write(f"{x_i}\n")


def plot_graph_connectivity(g) -> None:
    """ """
    G = nx.DiGraph()
    all_edges = []
    L_edges = []

    for eid in range(len(g["edges"])):
        edge = g["edges"][eid]

        # convert 1-indexed to 0-indexed
        i = int(edge["fromIdx"]) - 1
        j = int(edge["toIdx"]) - 1

        all_edges += [(i, j)]
        edge_type = edge["type"][0][0]
        if edge_type == "L":
            L_edges += [(i, j)]

    G.add_edges_from(all_edges)

    # values = [val_map.get(node, 0.25) for node in G.nodes()]

    # Specify the edges you want here
    # edge_colours = ['black' if not edge in L_edges else 'red' for edge in G.edges()]
    P_edges = [edge for edge in G.edges() if edge not in L_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    # pos = nx.spring_layout(G)

    pos = nx.kamada_kawai_layout(G)

    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap("jet"), node_size=50)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=P_edges, edge_color="k", arrows=False)
    nx.draw_networkx_edges(G, pos, edgelist=L_edges, edge_color="r", arrows=True)
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
