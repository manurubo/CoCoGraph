from lib_functions.config import *
from lib_functions.adjacency_utils import pad_adjs, count_bond_types
import numpy as np
import networkx as nx
from rdkit import Chem
import itertools
from lib_functions.adjacency_utils import nx_to_rdkit
from itertools import permutations, combinations
import pandas as pd
import scipy

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# Convert rdkit molecule to graph
def mol_to_graph(mol):
    
    # Change aromatic bonds to single/double
    Chem.rdmolops.Kekulize(mol)
    
    # Add this line to check for dummy atoms
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None
    
    # Initialize the graph
    G = nx.MultiGraph(directed=False)
    # Build id dictionaries
    atom_count = {'H' : 0}
    mol2graph = {}
    # Add atoms
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol not in atom_count:
            atom_count[atom_symbol] = 0
        atom_count[atom_symbol] += 1
        glabel = '%s_%d' % (atom_symbol, atom_count[atom_symbol])
        mol2graph[atom.GetIdx()] = glabel
        G.add_node(
            glabel,
            label=atom_symbol,
            atomic_num=atom.GetAtomicNum(),
            formal_charge = atom.GetFormalCharge()
        )
        
        #print(atom.GetNumImplicitHs())
        #total_hydrogens = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
        total_hydrogens = atom.GetNumImplicitHs()
        for nh in range(total_hydrogens):
            atom_count['H'] += 1
            glabelh = 'H_%d' % (atom_count['H'])
            G.add_node(
                glabelh,
                label='H',
                atomic_num=1,
                formal_charge = atom.GetFormalCharge()
            )
            G.add_edge(glabel, glabelh)           
    # Add bonds
    for bond in mol.GetBonds():
        if str(bond.GetBondType()) == 'SINGLE':
            nbond = 1
        elif str(bond.GetBondType()) == 'DOUBLE':
            nbond = 2
        elif str(bond.GetBondType()) == 'TRIPLE':
            nbond = 3
        else:
            print(bond.GetBondType(), file=sys.stderr)
            raise
        for nb in range(nbond):
            G.add_edge(
                mol2graph[bond.GetBeginAtomIdx()],
                mol2graph[bond.GetEndAtomIdx()],
            )
    # Done
    return G

# Convert SMILES to graph
def smiles_to_graph(sm):
    return mol_to_graph(Chem.MolFromSmiles(sm))

    
def embed_graph_nodes_norm(graph, encel=ENCEL, cycle_max=14):
    # Get things ready
    node_embeddings = []
    gsingle = nx.Graph(graph)
    #the_cycles = nx.cycle_basis(gsingle)
    ff = 1.
    the_cycles = [c for c in nx.cycle_basis(gsingle) if len(c) > 2]
    #ff = 2.

    # EMBED GRAPH
    # Cycles
    features = [0 for nc in range(1, cycle_max)]
    for cycle in the_cycles:
        try:
            features[len(cycle)-2] += 1  # Minimum loop is of size 3
        except IndexError: # First element is for loops of other lengths
            features[0] += 1

    # Calcular el total de ciclos
    total_cycles = sum(features)

    # Normalizar por el total de ciclos
    if total_cycles > 0:
        features = [count / total_cycles for count in features]

    # Planarity
    features.append(int(nx.is_planar(graph)))
    # Number of connected components (should be 1 for a real molecule, but could not be for randomizations)
    if nx.number_connected_components(graph) == 1:
         features.append(1)
    else: 
        features.append(0)
    
    # Fraction of bonds that are bridges (in the multigraph and in the simple graph)
    features.append(len([b for b in nx.bridges(graph)]) / graph.number_of_edges())
    features.append(len([b for b in nx.bridges(gsingle)]) / gsingle.number_of_edges())
    # Fraction of single, double, triple and quadruple edges
    multiedges = [0 for n in range(4)]
    norm = 2 * graph.number_of_edges()
    for n1 in graph.nodes():
        for n2 in graph.nodes():
            num = min(graph.number_of_edges(n1, n2),4)
            if num > 0:
                multiedges[num-1] += num/norm
    features += multiedges

    # Done with the graph
    gfeatures = torch.Tensor(features)
    # EMBED NODES

     # Bridge count for each node
    bridge_count = {node: 0 for node in graph.nodes()}
    puentes= nx.bridges(graph)
    for u, v in puentes:
        bridge_count[u] += 1
        bridge_count[v] += 1

    

    # print("aqui")
    # Loop over nodes
    node_list = []
    for node in graph.nodes():
        # Elements one-hot
        embedding = [0] * len(encel)
        element = node.split('_')[0]
        try:
            embedding[encel.index(element)] = 1
        except ValueError: # This atom is not in the encel list
            pass
       
        # In cycle
        in_cycle_size = [0] * (cycle_max-1)    # We distinguish between cycles of size 3-cycle_max and others
        for cycle in the_cycles:
            if node in cycle:
                cycsize = len(cycle)
                if cycsize > 2 and cycsize <= cycle_max:
                    in_cycle_size[cycsize - 3] += 1 # divide by 2 if symple_cycles
                else:
                    in_cycle_size[cycle_max-2] += 1 # divide by 2 if symple_cycles

        # Calcular el total de ciclos
        total_cycles = sum(in_cycle_size)
    
        # Normalizar por el total de ciclos
        if total_cycles > 0:
            in_cycle_size = [count / total_cycles for count in in_cycle_size]
        embedding += in_cycle_size
        # Distinct neighbors (heavy atom and hydrogens, respectively)
        neighbors = [0, 0]
        vecinos = len(list(gsingle.neighbors(node)))
        for ne in gsingle.neighbors(node):
            if ne.startswith('H_'):
                neighbors[1] += 1/vecinos
            else:
                neighbors[0] += 1/vecinos
        embedding += neighbors
        
         # Add bridge count for the node
        embedding.append(bridge_count[node]/vecinos)
        # Done with this node
        node_embeddings.append(embedding)
        node_list.append(node)
        
    
    node_embeddings = np.array(node_embeddings)
    node_embeddings = np.concatenate([node_embeddings,
                                            np.zeros([MAX_ATOM - node_embeddings.shape[0],
                                                      node_embeddings.shape[1]])],
                                           axis=0)
    # Done with all nodes
    nfeatures = torch.Tensor(node_embeddings)

    # Calculate distances between every pair of nodes efficiently
    all_pairs_shortest_path_length = dict(nx.shortest_path_length(gsingle))

    # One-hot encoding for distances with an additional category for 'no path'
    num_nodes = len(node_list)
    one_hot_distances = np.zeros((MAX_ATOM, MAX_ATOM, 12))  # 12 categories (1-10, >10, no path)
    # print("casi")
    for i, node1 in enumerate(node_list):
        for j, node2 in enumerate(node_list):
            if node1 in all_pairs_shortest_path_length and node2 in all_pairs_shortest_path_length[node1]:
                distance = all_pairs_shortest_path_length[node1][node2]
                if distance > 10:
                    one_hot_distances[i, j, 10] = 1  # Index 10 for distances > 10
                else:
                    one_hot_distances[i, j, distance - 1] = 1  # Index distance - 1 for distances 1-10
            else:
                one_hot_distances[i, j, 11] = 1  # Index 11 for 'no path'

    return gfeatures, nfeatures, one_hot_distances

def embed_graph_nodes_norm_timepred(graph, encel=ENCEL, cycle_max=14):
    # Get things ready
    node_embeddings = []
    gsingle = nx.Graph(graph)
    #the_cycles = nx.cycle_basis(gsingle)
    ff = 1.
    the_cycles = [c for c in nx.cycle_basis(gsingle) if len(c) > 2]
    #ff = 2.

    # EMBED GRAPH
    # Cycles
    features = [0 for nc in range(1, cycle_max)]
    for cycle in the_cycles:
        try:
            features[len(cycle)-2] += 1  # Minimum loop is of size 3
        except IndexError: # First element is for loops of other lengths
            features[0] += 1

    # Calcular el total de ciclos
    total_cycles = sum(features)

    # Normalizar por el total de ciclos
    if total_cycles > 0:
        features = [count / total_cycles for count in features]

    # Planarity
    features.append(int(nx.is_planar(graph)))
    # Number of connected components (should be 1 for a real molecule, but could not be for randomizations)
    if nx.number_connected_components(graph) == 1:
         features.append(1)
    else: 
        features.append(0)
    
    # Fraction of bonds that are bridges (in the multigraph and in the simple graph)
    features.append(len([b for b in nx.bridges(graph)]) / graph.number_of_edges())
    features.append(len([b for b in nx.bridges(gsingle)]) / gsingle.number_of_edges())
    # Fraction of single, double, triple and quadruple edges
    multiedges = [0 for n in range(4)]
    norm = 2 * graph.number_of_edges()
    for n1 in graph.nodes():
        for n2 in graph.nodes():
            num = min(graph.number_of_edges(n1, n2),4)
            if num > 0:
                multiedges[num-1] += num/norm
    features += multiedges
    
    # Done with the graph
    gfeatures = torch.Tensor(features)
    
    
    # DONE
    return gfeatures

def embed_edges_manuel(g,node_list):
    adj = nx.to_numpy_array(g, nodelist=node_list)
    bonds_num, num_atoms = count_bond_types(adj) 
    pad_adj  = pad_adjs(adj, node_number=MAX_ATOM)
    emb = torch.Tensor(pad_adj)
    return emb, bonds_num, num_atoms


import numpy as np
from datetime import datetime

# STAY
def all_simple_cycles(G):
    # Convertir el grafo no dirigido en dirigido
    gsingle = nx.Graph(G)
    DG = gsingle.to_directed()

    ciclos = nx.simple_cycles(DG)

    return ciclos


# STAY  
def create_cycle_dict(cycle_basis, cycle_max=14):
    # Crear un diccionario para mapear cada enlace a los ciclos y sus tamaños en los que se encuentra.
    edge_to_cycle_sizes = {}
    cycles_total = []
    for cycle in cycle_basis:
        cycle_size = len(cycle)  # Tamaño del ciclo
        cycles_total.append(cycle)
        if cycle_size>2:
            if cycle_size >cycle_max:
                cycle_size=cycle_max
            # Crear pares de enlaces para cada ciclo (asumiendo enlaces no dirigidos)
            edges_in_cycle = [(cycle[j], cycle[(j + 1) % cycle_size]) for j in range(cycle_size)]
            for edge in edges_in_cycle:
                if edge not in edge_to_cycle_sizes:
                    edge_to_cycle_sizes[edge] = []
                if (edge[1], edge[0]) not in edge_to_cycle_sizes:
                    edge_to_cycle_sizes[(edge[1], edge[0])] = []
                edge_to_cycle_sizes[edge].append(cycle_size)
                edge_to_cycle_sizes[(edge[1], edge[0])].append(cycle_size)

    return edge_to_cycle_sizes, cycles_total

#STAY
def find_edge_cycles_with_sizes(g,simple = True, cycle_max=14):
    # Encontrar una base de ciclos para el grafo.
    if simple:
        cycle_basis = all_simple_cycles(g)
    else:
        cycle_basis = nx.cycle_basis(nx.Graph(g))


    edge_to_cycle_sizes, cycles_total = create_cycle_dict(cycle_basis)

    return edge_to_cycle_sizes, cycles_total

# STAY
def embed_edges_with_cycle_sizes_norm(g, cycle_max=14):
    
    # Obtener los nodos
    nodes = list(g.nodes())

    # Obtener los enlaces y sus atributos (one-hot para SINGLE, DOUBLE, TRIPLE)
    edges, edge_attributes = [], []
    
    # Encuentra los ciclos y sus tamaños en los que cada enlace está involucrado
    edge_to_cycle_sizes, cycles_total = find_edge_cycles_with_sizes(g, simple=False)

    # Identify bridges
    bridges = set(nx.bridges(g))
#     print(edge_to_cycle_sizes)
    for n1, n2 in g.edges():
        nn1, nn2 = nodes.index(n1), nodes.index(n2)
        # Asegurarse de que cada enlace se añade una sola vez
        if (nn1, nn2) not in edges and (nn2, nn1) not in edges:

            # Multiplicidad del enlace
            bond_type = g.number_of_edges(n1, n2) - 1
            # Tamaños de los ciclos para el enlace actual (3-14)
            cycle_sizes = edge_to_cycle_sizes.get((n1, n2), []) + edge_to_cycle_sizes.get((n2, n1), [])
            cycle_size_attributes = [cycle_sizes.count(i)/2 if i in cycle_sizes else 0 for i in range(3, cycle_max+1)]

            # Calcular el total de ciclos
            total_cycles = sum(cycle_size_attributes)
        
            # Normalizar por el total de ciclos
            if total_cycles > 0:
                cycle_size_attributes = [count / total_cycles for count in cycle_size_attributes]

            # Check if the edge is a bridge
            is_bridge = 1 if (n1, n2) in bridges or (n2, n1) in bridges else 0


            
            # Calculate the number of distinct paths between n1 and n2 (with optional cutoff
            num_paths = len(list(nx.all_simple_paths(nx.Graph(g), n1, n2, cutoff=10)))
            num_paths = np.log1p(num_paths) / np.log1p(150)
            

            # Añadir atributos del enlace, incluyendo la información del ciclo
            edge_attr = [1 if i == bond_type else 0 for i in range(3)] + cycle_size_attributes + [is_bridge] + [num_paths]
            # Añadir enlace (directo)
            edges.append((nn1, nn2))
            edge_attributes.append(edge_attr)
            # Añadir enlace (inverso)
            edges.append((nn2, nn1))
            edge_attributes.append(edge_attr)
    
    
    # Convertir a tensores de PyTorch
    return torch.LongTensor(edges).transpose(1, 0), torch.Tensor(edge_attributes)




 


from rdkit.Chem import AllChem


def calculate_2d_distances_ordered(graph, original_node_order):
    # Convert NetworkX graph to RDKit molecule (including hydrogens)
    mol = nx_to_rdkit(graph, hidrogenos=True)
    
    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Get the original node order from the graph
    # original_node_order = list(graph.nodes())
    
    # Create a mapping from original node names to RDKit atom indices
    node_to_atom_index = {node: mol.GetAtomWithIdx(i).GetIdx() for i, node in enumerate(original_node_order)}
    
    # Get number of atoms
    num_atoms = mol.GetNumAtoms()
    
    # Create a matrix to store distances
    distances = np.zeros((MAX_ATOM, MAX_ATOM))
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Calculate pairwise distances
    for i, node_i in enumerate(original_node_order):
        atom_i = node_to_atom_index[node_i]
        pos_i = conf.GetAtomPosition(atom_i)
        for j, node_j in enumerate(original_node_order):
            atom_j = node_to_atom_index[node_j]
            pos_j = conf.GetAtomPosition(atom_j)
            # Calculate Euclidean distance in 2D
            dist = ((pos_i.x - pos_j.x)**2 + (pos_i.y - pos_j.y)**2)**0.5
            # Store distance
            distances[i, j] = dist

    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    if std_dist != 0:
        normalized_distances = (distances - mean_dist) / std_dist
    else:
        normalized_distances = distances  # If all distances are the same
    
    return normalized_distances

def count_cycles_by_size(graph, max_size=14):
    cycle_basis = nx.cycle_basis(graph)
    cycle_counts = torch.zeros(max_size+1-3)

    # print(cycle_basis)
    for cycle in cycle_basis:
        size = len(cycle)
        # print(cycle, size)
        if size <= max_size:
            cycle_counts[size-3] += 1
        else:
            cycle_counts[max_size-3] += 1  # Group all cycles larger than max_size
    # print(cycle_counts)
    return cycle_counts


