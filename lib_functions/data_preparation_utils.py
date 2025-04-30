from lib_functions.config import *
from lib_functions.libraries import *
from lib_functions.adjacency_utils import  pad_adjs, count_bond_types
from lib_functions.adjacency_utils import nx_to_rdkit

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from lib_functions.adjacency_utils import genera_intermedio

from rdkit.Chem import AllChem
import json




# Convert rdkit molecule to graph
def mol_to_graph(mol):
    """Converts an RDKit molecule object into a NetworkX MultiGraph.

    Handles implicit hydrogens and kekulizes aromatic bonds.

    Args:
        mol (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        networkx.MultiGraph or None: The corresponding NetworkX graph, or None 
                                     if the molecule contains dummy atoms.
    """
    
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
        
        # if we want to include hydrogens explicitly
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
    """Converts a SMILES string to a NetworkX MultiGraph.

    Args:
        sm (str): The SMILES string representation of the molecule.

    Returns:
        networkx.MultiGraph: The corresponding NetworkX graph.
    """
    return mol_to_graph(Chem.MolFromSmiles(sm))

    
def embed_graph_nodes_norm(graph, encel=ENCEL, cycle_max=14):
    """Embeds graph-level and node-level features for a given graph.

    Graph features include cycle counts (normalized), planarity, connectivity,
    bridge fraction, and bond type fractions.
    Node features include one-hot element encoding, cycle membership (normalized),
    neighbor types (heavy atom/hydrogen fraction), and bridge count (normalized).
    Also calculates one-hot encoded shortest path distances between all node pairs.

    Args:
        graph (networkx.MultiGraph): The input graph.
        encel (list, optional): List of element symbols for one-hot encoding.
                                Defaults to ENCEL.
        cycle_max (int, optional): Maximum cycle size to consider explicitly.
                                 Defaults to 14.

    Returns:
        tuple: 
            - torch.Tensor: Graph-level feature tensor.
            - torch.Tensor: Padded node-level feature tensor (MAX_ATOM x num_node_features).
            - np.ndarray: Padded one-hot encoded shortest path distance matrix 
                          (MAX_ATOM x MAX_ATOM x 12).
    """
    # Get things ready
    node_embeddings = []
    gsingle = nx.Graph(graph)
    the_cycles = [c for c in nx.cycle_basis(gsingle) if len(c) > 2]

    # EMBED GRAPH
    # Cycles
    features = [0 for nc in range(1, cycle_max)]
    for cycle in the_cycles:
        try:
            features[len(cycle)-2] += 1  # Minimum loop is of size 3
        except IndexError: # First element is for loops of other lengths
            features[0] += 1

    # Calculate the total number of cycles
    total_cycles = sum(features)

    # Normalize by the total number of cycles
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

        # Calculate the total number of cycles
        total_cycles = sum(in_cycle_size)
    
        # Normalize by the total number of cycles
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
    """Embeds graph-level features for time prediction tasks.

    Similar to embed_graph_nodes_norm but only calculates and returns graph-level features.

    Args:
        graph (networkx.MultiGraph): The input graph.
        encel (list, optional): List of element symbols (unused in this function).
                                Defaults to ENCEL.
        cycle_max (int, optional): Maximum cycle size to consider explicitly.
                                 Defaults to 14.

    Returns:
        torch.Tensor: Graph-level feature tensor.
    """
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
    """Generates a padded adjacency matrix representation of the graph.

    Also counts bond types and the number of atoms.

    Args:
        g (networkx.Graph): The input graph.
        node_list (list): The ordered list of nodes to use for the adjacency matrix.

    Returns:
        tuple:
            - torch.Tensor: Padded adjacency matrix (MAX_ATOM x MAX_ATOM).
            - np.ndarray: Counts of single, double, triple, quadruple bonds.
            - int: Number of atoms in the graph.
    """
    adj = nx.to_numpy_array(g, nodelist=node_list)
    bonds_num, num_atoms = count_bond_types(adj) 
    pad_adj  = pad_adjs(adj, node_number=MAX_ATOM)
    emb = torch.Tensor(pad_adj)
    return emb, bonds_num, num_atoms


def all_simple_cycles(G):
    """Finds all simple cycles in a NetworkX graph.

    Converts the graph to a directed graph to use nx.simple_cycles.

    Args:
        G (networkx.Graph): The input graph.

    Returns:
        generator: A generator yielding lists of nodes representing simple cycles.
    """
    gsingle = nx.Graph(G)
    DG = gsingle.to_directed()

    ciclos = nx.simple_cycles(DG)

    return ciclos


def create_cycle_dict(cycle_basis, cycle_max=14):
    """Creates a dictionary mapping edges to the sizes of cycles they belong to.

    Args:
        cycle_basis (iterable): An iterable of cycles (lists of nodes).
        cycle_max (int, optional): Maximum cycle size to record. Cycles larger
                                 than this are recorded as size cycle_max.
                                 Defaults to 14.

    Returns:
        tuple:
            - dict: A dictionary where keys are edge tuples (u, v) and values are
                    lists of cycle sizes the edge belongs to.
            - list: The input cycle_basis converted to a list.
    """
    # Create a dictionary to map each edge to the cycles and their sizes in which it is found.
    edge_to_cycle_sizes = {}
    cycles_total = []
    for cycle in cycle_basis:
        cycle_size = len(cycle)  # Size of the cycle
        cycles_total.append(cycle)
        if cycle_size>2:
            if cycle_size >cycle_max:
                cycle_size=cycle_max
            # Create pairs of edges for each cycle (assuming undirected edges)
            edges_in_cycle = [(cycle[j], cycle[(j + 1) % cycle_size]) for j in range(cycle_size)]
            for edge in edges_in_cycle:
                if edge not in edge_to_cycle_sizes:
                    edge_to_cycle_sizes[edge] = []
                if (edge[1], edge[0]) not in edge_to_cycle_sizes:
                    edge_to_cycle_sizes[(edge[1], edge[0])] = []
                edge_to_cycle_sizes[edge].append(cycle_size)
                edge_to_cycle_sizes[(edge[1], edge[0])].append(cycle_size)

    return edge_to_cycle_sizes, cycles_total

def find_edge_cycles_with_sizes(g,simple = True, cycle_max=14):
    """Finds cycles in a graph and maps edges to the sizes of cycles they belong to.

    Args:
        g (networkx.Graph): The input graph.
        simple (bool, optional): Whether to find simple cycles (True) or use 
                                 nx.cycle_basis (False). Defaults to True.
        cycle_max (int, optional): Maximum cycle size for the cycle dictionary.
                                 Defaults to 14.

    Returns:
        tuple:
            - dict: Dictionary mapping edges to lists of cycle sizes.
            - list: List of all cycles found.
    """
    # Find a cycle basis for the graph.
    if simple:
        cycle_basis = all_simple_cycles(g)
    else:
        cycle_basis = nx.cycle_basis(nx.Graph(g))


    edge_to_cycle_sizes, cycles_total = create_cycle_dict(cycle_basis)

    return edge_to_cycle_sizes, cycles_total

def embed_edges_with_cycle_sizes_norm(g, cycle_max=14):
    """Embeds edge features including bond type, normalized cycle membership, 
    bridge status, and normalized path count.

    Args:
        g (networkx.MultiGraph): The input graph.
        cycle_max (int, optional): Maximum cycle size to consider explicitly for features.
                                 Defaults to 14.

    Returns:
        tuple:
            - torch.LongTensor: Edge index tensor of shape (2, num_edges * 2).
            - torch.Tensor: Edge attribute tensor of shape (num_edges * 2, num_features).
                          Features: [bond_type_1hot (3), cycle_sizes (cycle_max-2),
                                     is_bridge (1), log_num_paths (1)].
    """
    
    # Get the nodes
    nodes = list(g.nodes())

    # Get the edges and their attributes (one-hot for SINGLE, DOUBLE, TRIPLE)
    edges, edge_attributes = [], []
    
    # Find the cycles and their sizes in which each edge is involved
    edge_to_cycle_sizes, cycles_total = find_edge_cycles_with_sizes(g, simple=False)

    # Identify bridges
    bridges = set(nx.bridges(g))
    for n1, n2 in g.edges():
        nn1, nn2 = nodes.index(n1), nodes.index(n2)
        # Ensure each edge is added only once
        if (nn1, nn2) not in edges and (nn2, nn1) not in edges:

            # Multiplicity of the edge  
            bond_type = g.number_of_edges(n1, n2) - 1
            # Sizes of the cycles for the current edge (3-14)
            cycle_sizes = edge_to_cycle_sizes.get((n1, n2), []) + edge_to_cycle_sizes.get((n2, n1), [])
            cycle_size_attributes = [cycle_sizes.count(i)/2 if i in cycle_sizes else 0 for i in range(3, cycle_max+1)]

            # Calculate the total number of cycles
            total_cycles = sum(cycle_size_attributes)
        
            # Normalize by the total number of cycles
            if total_cycles > 0:
                cycle_size_attributes = [count / total_cycles for count in cycle_size_attributes]

            # Check if the edge is a bridge
            is_bridge = 1 if (n1, n2) in bridges or (n2, n1) in bridges else 0

            # Calculate the number of distinct paths between n1 and n2 (with optional cutoff
            num_paths = len(list(nx.all_simple_paths(nx.Graph(g), n1, n2, cutoff=10)))
            num_paths = np.log1p(num_paths) / np.log1p(150)
            
            # Add edge attributes, including cycle information
            edge_attr = [1 if i == bond_type else 0 for i in range(3)] + cycle_size_attributes + [is_bridge] + [num_paths]
            # Add edge (direct)
            edges.append((nn1, nn2))
            edge_attributes.append(edge_attr)
            # Add edge (inverse)
            edges.append((nn2, nn1))
            edge_attributes.append(edge_attr)
    
    
    # Convert to PyTorch tensors
    return torch.LongTensor(edges).transpose(1, 0), torch.Tensor(edge_attributes)




def calculate_2d_distances_ordered(graph, original_node_order):
    """Calculates pairwise 2D Euclidean distances between atoms in a graph.

    Uses RDKit to generate 2D coordinates and preserves the original node order.
    Distances are normalized (z-score).

    Args:
        graph (networkx.Graph): The input graph.
        original_node_order (list): The list of nodes in the desired order.

    Returns:
        np.ndarray: A padded (MAX_ATOM x MAX_ATOM) matrix of normalized 
                    pairwise 2D distances.
    """
    # Convert NetworkX graph to RDKit molecule (including hydrogens)
    mol = nx_to_rdkit(graph, hidrogenos=True)
    
    # Compute 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    
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
    """Counts cycles in a graph based on their size using nx.cycle_basis.

    Args:
        graph (networkx.Graph): The input graph.
        max_size (int, optional): The maximum cycle size to count individually.
                                Larger cycles are grouped together. Defaults to 14.

    Returns:
        torch.Tensor: A tensor where index i corresponds to the count of 
                      cycles of size i+3. The last element counts cycles > max_size.
    """
    cycle_basis = nx.cycle_basis(graph)
    cycle_counts = torch.zeros(max_size+1-3)

    for cycle in cycle_basis:
        size = len(cycle)
        if size <= max_size:
            cycle_counts[size-3] += 1
        else:
            cycle_counts[max_size-3] += 1  # Group all cycles larger than max_size
    return cycle_counts

def compute_features(graph, num, swaps_to_undo):
    """Computes various graph, node, and edge features for a potentially modified graph.
    
    Applies swaps defined in swaps_to_undo before computing features.
    Features include: padded adjacency, graph embeddings, node embeddings, 
    distance matrix, edge index, edge attributes, atom count, and 2D positions.

    Args:
        graph (networkx.MultiGraph): The base input graph.
        num (int): An index or identifier for this computation.
        swaps_to_undo (list): List of edge swap tuples (u, v, x, y) to apply.
        
    Returns:
        tuple: 
            - ruido (torch.Tensor): Padded adjacency matrix.
            - gemb (torch.Tensor): Graph-level embeddings.
            - nemb (torch.Tensor): Padded node-level embeddings.
            - distances (np.ndarray): Padded one-hot shortest path distance matrix.
            - edge_index (torch.LongTensor): Edge index tensor.
            - edge_attr (torch.Tensor): Edge attribute tensor.
            - natoms (int): Number of atoms.
            - num (int): The input index number.
            - dosd_positions (np.ndarray): Padded normalized 2D distance matrix.
    """
    
    grafo_i = genera_intermedio(graph, swaps_to_undo)
    ruido, _, natoms = embed_edges_manuel(grafo_i, list(grafo_i.nodes()))
    gemb, nemb, distances = embed_graph_nodes_norm(grafo_i)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(grafo_i)
    dosd_positions = calculate_2d_distances_ordered(grafo_i, list(grafo_i.nodes()))
    
    del(grafo_i)
    return ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd_positions

def compute_features_cero(grafo_i):
    """Computes various graph, node, and edge features for a given graph (no swaps).
    
    Features include: padded adjacency, graph embeddings, node embeddings, 
    distance matrix, edge index, edge attributes, atom count, and 2D positions.

    Args:
        grafo_i (networkx.MultiGraph): The input graph.
        
    Returns:
        tuple: 
            - ruido (torch.Tensor): Padded adjacency matrix.
            - gemb (torch.Tensor): Graph-level embeddings.
            - nemb (torch.Tensor): Padded node-level embeddings.
            - distances (np.ndarray): Padded one-hot shortest path distance matrix.
            - edge_index (torch.LongTensor): Edge index tensor.
            - edge_attr (torch.Tensor): Edge attribute tensor.
            - natoms (int): Number of atoms.
            - num (int): Always returns 0.
            - dosd_positions (np.ndarray): Padded normalized 2D distance matrix.
    """
    ruido, _, natoms = embed_edges_manuel(grafo_i, list(grafo_i.nodes()))
    gemb, nemb, distances = embed_graph_nodes_norm(grafo_i)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(grafo_i)
    dosd_positions = calculate_2d_distances_ordered(grafo_i, list(grafo_i.nodes()))
    
    return ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, 0, dosd_positions

def compute_features_timepred(graph, num, swaps_to_undo):
    """Computes graph-level features for time prediction after applying swaps.
    
    Args:
        graph (networkx.MultiGraph): The base input graph.
        num (int): An index or identifier (unused in return).
        swaps_to_undo (list): List of edge swap tuples (u, v, x, y) to apply.
        
    Returns:
        torch.Tensor: Graph-level embeddings (gemb).
    """
    
    grafo_i = genera_intermedio(graph, swaps_to_undo)
    gemb = embed_graph_nodes_norm_timepred(grafo_i)
    
    del(grafo_i)
    return gemb

def save_plot_data(data, filename):
    """Saves data to a JSON file.
    
    Args:
        data: The data structure (e.g., dict, list) to save.
        filename (str): Path to the output JSON file.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

def generate_swap_tensors_optimized(final_swaps, num_nodes=MAX_ATOM, device=device):
    """Generates a tensor representing edge swaps in a 4D format, optimized for GPU.
    
    Creates a tensor where swap (u, v, x, y) is marked by setting specific 
    4D indices to 1.
    
    Args:
        final_swaps (list): List of tuples (u, v, x, y) representing swaps.
                           Indices u, v, x, y should be node indices.
        num_nodes (int, optional): Total number of nodes (dimension size).
                                 Defaults to MAX_ATOM.
        device (torch.device, optional): Device to allocate the tensor on.
                                        Defaults to the global `device`.
        
    Returns:
        torch.Tensor: Tensor of shape [len(final_swaps), num_nodes, num_nodes, 
                      num_nodes, num_nodes] representing the swaps.
    """
    # Preallocate tensor on the specified device
    all_swaps_tensor = torch.zeros((len(final_swaps), num_nodes, num_nodes, num_nodes, num_nodes), device=device)

    for idx, swap in enumerate(final_swaps):
        u, v, x, y = swap

        # Perform operations directly in the preallocated tensor
        all_swaps_tensor[idx, u, x, v, y] = 1
        all_swaps_tensor[idx, x, u, y, v] = 1
        all_swaps_tensor[idx, v, y, u, x] = 1
        all_swaps_tensor[idx, y, v, x, u] = 1

    return all_swaps_tensor 


def compute_features_fps(graph, num, swaps_to_undo):
    """Computes various graph features including Morgan fingerprints after applying swaps.

    Similar to compute_features, but also calculates and returns Morgan fingerprints.

    Args:
        graph (networkx.MultiGraph): The base input graph.
        num (int): An index or identifier for this computation.
        swaps_to_undo (list): List of edge swap tuples (u, v, x, y) to apply.

    Returns:
        tuple:
            - ruido (torch.Tensor): Padded adjacency matrix.
            - gemb (torch.Tensor): Graph-level embeddings.
            - nemb (torch.Tensor): Padded node-level embeddings.
            - distances (np.ndarray): Padded one-hot shortest path distance matrix.
            - edge_index (torch.LongTensor): Edge index tensor.
            - edge_attr (torch.Tensor): Edge attribute tensor.
            - natoms (int): Number of atoms.
            - num (int): The input index number.
            - dosd_positions (np.ndarray): Padded normalized 2D distance matrix.
            - fingerprint (list): Morgan fingerprint as a list of bits.
    """

    grafo_i = genera_intermedio(graph,swaps_to_undo)
    ruido, _, natoms = embed_edges_manuel(grafo_i, list(grafo_i.nodes()))
    gemb, nemb, distances = embed_graph_nodes_norm(grafo_i)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(grafo_i)
    dosd_positions = calculate_2d_distances_ordered(grafo_i, list(grafo_i.nodes())) 

    mol = nx_to_rdkit(grafo_i, False)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)
    fingerprint = list(fingerprint)
    del(grafo_i)

    return ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd_positions, fingerprint

def compute_features_cero_fps(grafo_i):
    """Computes various graph features including Morgan fingerprints (no swaps).

    Similar to compute_features_cero, but also calculates and returns Morgan fingerprints.

    Args:
        grafo_i (networkx.MultiGraph): The input graph.

    Returns:
        tuple:
            - ruido (torch.Tensor): Padded adjacency matrix.
            - gemb (torch.Tensor): Graph-level embeddings.
            - nemb (torch.Tensor): Padded node-level embeddings.
            - distances (np.ndarray): Padded one-hot shortest path distance matrix.
            - edge_index (torch.LongTensor): Edge index tensor.
            - edge_attr (torch.Tensor): Edge attribute tensor.
            - natoms (int): Number of atoms.
            - num (int): Always returns 0.
            - dosd_positions (np.ndarray): Padded normalized 2D distance matrix.
            - fingerprint (list): Morgan fingerprint as a list of bits.
    """

    ruido, _, natoms = embed_edges_manuel(grafo_i, list(grafo_i.nodes()))
    gemb, nemb, distances = embed_graph_nodes_norm(grafo_i)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(grafo_i)
    dosd_positions = calculate_2d_distances_ordered(grafo_i, list(grafo_i.nodes()))

    mol = nx_to_rdkit(grafo_i, False)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)
    fingerprint = list(fingerprint)
    
    return ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, 0, dosd_positions, fingerprint