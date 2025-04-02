from lib_functions.config import *
from lib_functions.libraries import *
import numpy as np
import networkx as nx
from rdkit import Chem
import torch
from copy import deepcopy


def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a


def components_to_graph(graph_nodes, adjacency):
    # Create an empty graph
    graph = nx.MultiGraph()
    
    adjacency = adjacency.triu(diagonal=1)

    graph.add_nodes_from(graph_nodes)
    lista_graph_nodes = list(graph_nodes)
    # Iterate over the adjacency matrix
    num_nodes = len(graph_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Check if there is an edge between nodes i and j
            if adjacency[i][j] != 0:
                edges = int(adjacency[i][j])
                # Add the edge to the graph
                for l in range(edges):
                    graph.add_edge(lista_graph_nodes[i][0], lista_graph_nodes[j][0])
    
    return graph

def generate_mask2(node_flags):
    return (node_flags > 0.9).unsqueeze(-1) * (node_flags > 0.9).unsqueeze(-2)

def count_bond_types(adj):
    no_bonds = np.count_nonzero(adj == 0)
    single_bonds = np.count_nonzero(adj == 1)
    double_bonds = np.count_nonzero(adj == 2)
    triple_bonds = np.count_nonzero(adj == 3)
    return [no_bonds-adj.shape[0], single_bonds, double_bonds, triple_bonds], adj.shape[0]


def round_half_up(x):
    return torch.floor(x + 0.5) 


def nx_to_rdkit(graph, hidrogenos = False):
    m = Chem.MolFromSmiles('')
    mw = Chem.RWMol(m)
    atom_index = {}
    for n, d in graph.nodes(data=True):
        atomo = Chem.Atom(d['label'])
        atomo.SetFormalCharge(d['formal_charge'])
        atom_index[n] = mw.AddAtom(atomo)
    for a, b, d in graph.edges(data=True):
        start = atom_index[a]
        end = atom_index[b]
        try:
            mw.AddBond(start, end, Chem.BondType.SINGLE)
        except RuntimeError: # Bond already exists: make it multiple
            bond = mw.GetBondBetweenAtoms(start, end)
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondType(Chem.BondType.DOUBLE)
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetBondType(Chem.BondType.TRIPLE)
            else:
                raise Exception('Quadruple bonds not allowed!')
    # Done
    mol = mw.GetMol()
    try:
        if not(hidrogenos):
            return Chem.RemoveHs(mol)
        else:
            return mol
    except Exception as e:
        print(f"Error in nx_to_rdkit: {e}")
        return mol


def connected_double_edge_swap(G, nswap=1, _window_threshold=3, seed=None):
    if not nx.is_connected(G):
        raise nx.NetworkXError("Graph not connected")
    if len(G) < 4:
        raise nx.NetworkXError("Graph has fewer than four nodes.")
        
    # Initialize the list to store intermediate graphs
    intermediate_graphs = []
    n = 0
    swapcount = 0
    deg = G.degree()
    # Label key for nodes
    dk = [n for n, d in G.degree()]
    cdf = nx.utils.cumulative_distribution([d for n, d in G.degree()])
    discrete_sequence = nx.utils.discrete_sequence
    window = 1
    final_swaps = []
    removed_edges_accumulated = []  # To accumulate removed edges
    added_edges_accumulated = []    # To accumulate added edges
    # Initialize temporary lists to hold the edges for accumulation
    temp_removed = []
    temp_added = []
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    while swapcount < nswap and n<100:
        wcount = 0
        swapped = []
        # If the window is small, we just check each time whether the graph is
        # connected by checking if the nodes that were just separated are still
        # connected.
        
        if window < _window_threshold:
            # This Boolean keeps track of whether there was a failure or not.
            fail = False
            while wcount < window and swapcount < nswap and n<100:
                # Pick two random edges without creating the edge list. Choose
                # source nodes from the discrete degree distribution.
                (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
                # If the source nodes are the same, skip this pair.
                if ui == xi:
                    continue
                # Convert an index to a node label.
                u = dk[ui]
                x = dk[xi]
                # Choose targets uniformly from neighbors.
                v = seed.choice(list(G.neighbors(u)))
                y = seed.choice(list(G.neighbors(x)))

                
                # If the target nodes are the same, skip this pair.
                if v == y or u ==y or x == v:
                    n += 1
                    continue
                if G.number_of_edges(u, x) > 2 or G.number_of_edges(v, y) > 2:
                    continue
                
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x)
                G.add_edge(v, y)
                swapped.append((u, v, x, y))
                swapcount += 1
                n += 1
                # If G remains connected...
                if nx.has_path(G, u, v):
                    wcount += 1
                    # Store the intermediate graph after successful swap
                    final_swaps.append((ui, node_to_index[v], xi, node_to_index[y]))
                    temp_removed.append([(ui,node_to_index[v]), (xi,node_to_index[y])])
                    temp_added.append([(ui, xi), (node_to_index[v], node_to_index[y])])
                    removed_edges_accumulated.append(list(temp_removed))
                    added_edges_accumulated.append(list(temp_added))
                # Otherwise, undo the changes.
                else:
                    G.add_edge(u, v)
                    G.add_edge(x, y)
                    G.remove_edge(u, x)
                    G.remove_edge(v, y)
                    swapcount -= 1
                    fail = True
            # If one of the swaps failed, reduce the window size.
            if fail:
                window = math.ceil(window / 2)
            else:
                window += 1
        # If the window is large, then there is a good chance that a bunch of
        # swaps will work. It's quicker to do all those swaps first and then
        # check if the graph remains connected.
        else:
            while wcount < window and n < 100 and swapcount < nswap:
                # Pick two random edges without creating the edge list. Choose
                # source nodes from the discrete degree distribution.
                (ui, xi) = discrete_sequence(2, cdistribution=cdf, seed=seed)
                # If the source nodes are the same, skip this pair.
                if ui == xi:
                    continue
                # Convert an index to a node label.
                u = dk[ui]
                x = dk[xi]
                # Choose targets uniformly from neighbors.
                v = seed.choice(list(G.neighbors(u)))
                y = seed.choice(list(G.neighbors(x)))
                # If the target nodes are the same, skip this pair.
                if v == y or u ==y or x == v:
                    n += 1
                    continue
                if G.number_of_edges(u, x) > 2 or G.number_of_edges(v, y) > 2:
                    continue
                G.remove_edge(u, v)
                G.remove_edge(x, y)
                G.add_edge(u, x)
                G.add_edge(v, y)
                swapped.append((u, v, x, y))
                swapcount += 1
                final_swaps.append((ui, node_to_index[v], xi, node_to_index[y]))
                temp_removed.append([(ui,node_to_index[v]), (xi,node_to_index[y])])
                temp_added.append([(ui, xi), (node_to_index[v], node_to_index[y])])
                removed_edges_accumulated.append(list(temp_removed))
                added_edges_accumulated.append(list(temp_added))
                n += 1
                wcount += 1
            # If the graph remains connected, increase the window size.
            if nx.is_connected(G):
                window += 1
            # Otherwise, undo the changes from the previous window and decrease
            # the window size.
            else:
                while swapped:
                    # print("revierte")
                    (u, v, x, y) = swapped.pop()
                    G.add_edge(u, v)
                    G.add_edge(x, y)
                    G.remove_edge(u, x)
                    G.remove_edge(v, y)
                    swapcount -= 1
                    final_swaps.pop()
                    temp_removed.pop()
                    temp_added.pop()
                    removed_edges_accumulated.pop()
                    added_edges_accumulated.pop()
                window = math.ceil(window / 2)
        
    if n == 100:
        end=True
    else:
        end = False
    return swapcount, G, intermediate_graphs, end, final_swaps, removed_edges_accumulated, added_edges_accumulated

def genera_intermedio(graph, deshacer_l):
    """
    Generates an intermediate graph by applying a series of double edge swaps.
    
    Args:
        graph: The input graph
        deshacer_l: List of tuples containing edges to swap
        
    Returns:
        Modified graph after applying the swaps
    """
    dk = [n for n, d in graph.degree()]
    for d in deshacer_l:
        u = dk[d[0][0]]
        v = dk[d[0][1]]
        x = dk[d[1][0]]
        y = dk[d[1][1]]
        graph.remove_edge(u, v)
        graph.remove_edge(x, y)
        graph.add_edge(u, x)
        graph.add_edge(v, y)
    return graph



