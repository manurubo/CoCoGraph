from lib_functions.config import *
from lib_functions.libraries import *
from rdkit import Chem


def pad_adjs(ori_adj, node_number):
    """
    Pads an adjacency matrix with zeros to match a target node number.

    Args:
        ori_adj (np.ndarray): The original adjacency matrix.
        node_number (int): The target number of nodes.

    Returns:
        np.ndarray: The padded adjacency matrix.

    Raises:
        ValueError: If the original matrix size is larger than node_number.
    """
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
    """
    Creates a NetworkX MultiGraph from nodes and an adjacency matrix.

    Assumes the upper triangle of the adjacency matrix contains edge counts.

    Args:
        graph_nodes (list): A list of nodes, often including attributes like ('C', {'label': 'C'}).
        adjacency (np.ndarray): The adjacency matrix where entries represent edge counts.

    Returns:
        nx.MultiGraph: The resulting graph.
    """
    # Create an empty graph
    graph = nx.MultiGraph()
    
    adjacency = adjacency.triu(diagonal=1) # Only upper triangle

    graph.add_nodes_from(graph_nodes) # Add nodes to the graph
    lista_graph_nodes = list(graph_nodes) # Convert to list
    # Iterate over the adjacency matrix
    num_nodes = len(graph_nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Check if there is an edge between nodes i and j
            if adjacency[i][j] != 0:
                edges = int(adjacency[i][j]) # Get the number of edges
                # Add the edge to the graph
                for l in range(edges):
                    graph.add_edge(lista_graph_nodes[i][0], lista_graph_nodes[j][0])
    
    return graph

def generate_padding_mask(node_flags):
    """
    Generates a square mask based on node flags.

    Nodes with flags > 0.9 are considered present. The mask is True
    where both corresponding nodes are present.

    Args:
        node_flags (torch.Tensor): A 1D tensor of node flags.

    Returns:
        torch.Tensor: A square boolean mask tensor.
    """
    return (node_flags > 0.9).unsqueeze(-1) * (node_flags > 0.9).unsqueeze(-2)

def count_bond_types(adj):
    """
    Counts the number of different bond types in an adjacency matrix.

    Assumes bond types are represented by integer values (0, 1, 2, 3).

    Args:
        adj (np.ndarray): The adjacency matrix.

    Returns:
        tuple[list[int], int]: A tuple containing:
            - A list with counts: [no_bonds (excluding diagonal), single_bonds, double_bonds, triple_bonds].
            - The number of nodes (shape[0] of the matrix).
    """
    no_bonds = np.count_nonzero(adj == 0)
    single_bonds = np.count_nonzero(adj == 1)
    double_bonds = np.count_nonzero(adj == 2)
    triple_bonds = np.count_nonzero(adj == 3)
    return [no_bonds-adj.shape[0], single_bonds, double_bonds, triple_bonds], adj.shape[0]


def nx_to_rdkit(graph, hidrogenos = False):
    """
    Converts a NetworkX graph representing a molecule into an RDKit molecule object.

    Handles atom labels, formal charges, and increments bond orders for multiple edges
    between the same atoms (up to triple bonds).

    Args:
        graph (nx.Graph): Input NetworkX graph. Nodes must have 'label' (atomic symbol)
                          and 'formal_charge' attributes.
        hidrogenos (bool, optional): If True, keeps explicit hydrogens.
                                     If False (default), removes explicit hydrogens using Chem.RemoveHs.

    Returns:
        Chem.Mol: The RDKit molecule object. Returns the partially built molecule
                  if RDKit sanitization fails.
    """
    m = Chem.MolFromSmiles('')
    mw = Chem.RWMol(m)
    atom_index = {}
    # Iterate over the nodes in the graph
    for n, d in graph.nodes(data=True):
        atomo = Chem.Atom(d['label']) # Get the atom
        atomo.SetFormalCharge(d['formal_charge']) # Set the formal charge
        atom_index[n] = mw.AddAtom(atomo) # Add the atom to the molecule
    # Iterate over the edges in the graph
    for a, b, d in graph.edges(data=True):
        start = atom_index[a] # Get the start atom
        end = atom_index[b] # Get the end atom
        try:
            mw.AddBond(start, end, Chem.BondType.SINGLE) # Add the bond
        except RuntimeError: # Bond already exists: make it multiple
            bond = mw.GetBondBetweenAtoms(start, end) # Get the bond
            if bond.GetBondType() == Chem.BondType.SINGLE: # If the bond is single
                bond.SetBondType(Chem.BondType.DOUBLE) # Set the bond to double
            elif bond.GetBondType() == Chem.BondType.DOUBLE: # If the bond is double
                bond.SetBondType(Chem.BondType.TRIPLE) # Set the bond to triple
            else:
                raise Exception('Quadruple bonds not allowed!')
    # Done
    mol = mw.GetMol() # Get the molecule
    try:
        if not(hidrogenos): # If hydrogens are not kept
            return Chem.RemoveHs(mol) # Remove the hydrogens
        else:
            return mol # Return the molecule
    except Exception as e:
        print(f"Error in nx_to_rdkit: {e}")    
        return mol


def connected_double_edge_swap(G, nswap=1, _window_threshold=3, seed=None):
    """
    Performs connected double edge swaps on a graph while maintaining connectivity.
    Modified from networkx.algorithms.swap.double_edge_swap

    Attempts to perform `nswap` swaps. A swap involves choosing two edges (u, v) and (x, y),
    removing them, and adding edges (u, x) and (v, y), provided connectivity is maintained
    and constraints (like no parallel edges beyond triple bonds) are met. Uses a windowed
    approach for efficiency.

    Args:
        G (nx.Graph): The input graph (must be connected and have >= 4 nodes).
        nswap (int, optional): The target number of successful swaps. Defaults to 1.
        _window_threshold (int, optional): Threshold for adapting the swap window size. Defaults to 3.
        seed (np.random.Generator | int | None, optional): Random seed or generator for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - swapcount (int): The actual number of successful swaps performed.
            - G (nx.Graph): The modified graph.
            - intermediate_graphs (list): Currently unused, returns an empty list.
            - end (bool): True if the maximum iteration limit was reached, False otherwise.
            - final_swaps (list[tuple]): List of tuples representing successful swaps (using original node indices).
            - removed_edges_accumulated (list[list[tuple]]): List storing lists of removed edges (indices) at each successful swap step.
            - added_edges_accumulated (list[list[tuple]]): List storing lists of added edges (indices) at each successful swap step.

    Raises:
        nx.NetworkXError: If the graph is not connected or has fewer than four nodes.
    """
    if not nx.is_connected(G): # If the graph is not connected
        raise nx.NetworkXError("Graph not connected") # Raise an error
    if len(G) < 4: # If the graph has fewer than four nodes
        raise nx.NetworkXError("Graph has fewer than four nodes.") # Raise an error
        
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
    final_swaps = [] # List to store the final swaps
    removed_edges_accumulated = [] # List to accumulate removed edges
    added_edges_accumulated = [] # List to accumulate added edges
    # Initialize temporary lists to hold the edges for accumulation
    temp_removed = []
    temp_added = []
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())} # Dictionary to map nodes to indices
    while swapcount < nswap and n<100: # While the number of swaps is less than the target and the number of tries is less than 100
        wcount = 0 # Counter for the number of successful swaps
        swapped = [] # List to store the swapped edges
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
    # If the number of tries is 100, set the end to True
    if n == 100:
        end=True # This marks that not all swaps were performed
    else:
        end = False # This marks that all swaps were performed  
    return swapcount, G, intermediate_graphs, end, final_swaps, removed_edges_accumulated, added_edges_accumulated

def genera_intermedio(graph, swaps_to_undo):
    """
    Generates an intermediate graph by applying a series of double edge swaps.
    
    Args:
        graph: The input graph
        swaps_to_undo: List of tuples containing edges to swap
        
    Returns:
        Modified graph after applying the swaps
    """
    dk = [n for n, d in graph.degree()]
    for d in swaps_to_undo:
        u = dk[d[0][0]]
        v = dk[d[0][1]]
        x = dk[d[1][0]]
        y = dk[d[1][1]]
        graph.remove_edge(u, v)
        graph.remove_edge(x, y)
        graph.add_edge(u, x)
        graph.add_edge(v, y)
    return graph



