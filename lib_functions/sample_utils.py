from lib_functions.adjacency_utils import components_to_graph, nx_to_rdkit
from lib_functions.config import *
from lib_functions.libraries import *
from lib_functions.data_preparation_utils import embed_edges_manuel, embed_edges_with_cycle_sizes_norm, smiles_to_graph, embed_graph_nodes_norm, calculate_2d_distances_ordered
from rdkit.Chem import AllChem




def sample_positions(cumulative_distribution, shape):
    """
    Samples a 4D position index based on a flattened cumulative probability distribution.

    Args:
        cumulative_distribution (Tensor): A 1D tensor representing the cumulative
                                          probability distribution, normalized to sum to 1.
        shape (tuple): The original 4D shape of the probability tensor before flattening.

    Returns:
        tuple: A tuple containing:
            - position_4d (torch.Tensor): The sampled 4D index.
            - index (int): The corresponding 1D index in the flattened distribution.
            - error (bool): A flag indicating if the sampled index corresponds to the
                          last element (potentially an error or boundary case).
    """
    # Generate a random value between 0 and 1
    random_value = torch.rand(1).item()
    
    # Find the corresponding index in the cumulative distribution
    index = torch.searchsorted(cumulative_distribution, random_value).item()
    # Convert the 1D index to 4D indices
    position_4d = torch.unravel_index(torch.tensor(index), shape)

    error = False
    if index == 24010000: # if its the last element, it will be an error
        error = True
    
    return position_4d, index, error


# filling molecule: 'CN(CC(=O)O)C(=N)N ' Creatine 
# Filling values are used when the number of steps in sampling is greater than the actual swaps for that molecule
# Create filling values for all variables
processed_graph_filling = smiles_to_graph('CN(CC(=O)O)C(=N)N')

tensor_filling = embed_edges_manuel(processed_graph_filling, list(processed_graph_filling.nodes()))

componentes_ant_filling = nx.number_connected_components(processed_graph_filling)
mol_filling = nx_to_rdkit(processed_graph_filling, True)
mol_filling.UpdatePropertyCache()
fingerprint_filling = AllChem.GetMorganFingerprintAsBitVect(mol_filling, radius=3)
fingerprint_filling = list(fingerprint_filling)
fingerprint_filling = torch.Tensor(fingerprint_filling) 

gemb_filling, nemb_filling, distances_filling = embed_graph_nodes_norm(processed_graph_filling)

edge_index_filling, edge_attr_filling = embed_edges_with_cycle_sizes_norm(processed_graph_filling)

dosd_positions_filling = calculate_2d_distances_ordered(processed_graph_filling, list(processed_graph_filling.nodes())) 
dosd_positions_filling = torch.Tensor(dosd_positions_filling)    

def calculate_data_molecule(graph, tensor, num_swaps, current_swap):
    """
    Processes a graph represented by node data and an adjacency tensor to generate
    various molecular representations and features.

    If the current swap count exceeds the total number of swaps, it returns pre-calculated
    filling values (based on Creatine).

    Args:
        graph (nx.Graph): A NetworkX graph object with node data.
        tensor (torch.Tensor): An adjacency tensor representation of the graph.
        num_swaps (int): The total number of swap operations planned.
        current_swap (int): The index of the current swap operation.

    Returns:
        tuple: A tuple containing:
            - processed_graph (nx.Graph): The processed graph constructed from nodes and tensor.
            - tensor (torch.Tensor): The input adjacency tensor.
            - mol (rdkit.Chem.Mol): An RDKit molecule object.
            - gemb (torch.Tensor): Global graph embeddings.
            - nemb (torch.Tensor): Node embeddings.
            - distances (torch.Tensor): Pairwise node distances.
            - edge_index (torch.Tensor): Edge indices for PyG.
            - edge_attr (torch.Tensor): Edge attributes for PyG.
            - dosd_positions (torch.Tensor): DOSD distances between nodes.
            - componentes_ant (int): Number of connected components in the graph.
    """

    #here I have to use the filling values
    if current_swap >= num_swaps:
        return processed_graph_filling, tensor_filling, mol_filling, gemb_filling, nemb_filling, distances_filling, edge_index_filling, edge_attr_filling, dosd_positions_filling, componentes_ant_filling

    processed_graph = components_to_graph(graph.nodes(data=True), tensor)
    componentes_ant = nx.number_connected_components(processed_graph)
    mol = nx_to_rdkit(processed_graph, True)
    gemb, nemb, distances = embed_graph_nodes_norm(processed_graph)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(processed_graph)
    
    dosd_positions = calculate_2d_distances_ordered(processed_graph, list(processed_graph.nodes())) 
    dosd_positions = torch.Tensor(dosd_positions)
    

    return processed_graph, tensor, mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, componentes_ant    

def calculate_data_molecule_fps(graph, tensor, num_swaps, current_swap):
    """
    Processes a graph represented by node data and an adjacency tensor to generate
    molecular representations, including Morgan fingerprints.

    Similar to `calculate_data_molecule`, but also calculates and returns Morgan
    fingerprints for the molecule.

    If the current swap count exceeds the total number of swaps, it returns pre-calculated
    filling values (based on Creatine).

    Args:
        graph (nx.Graph): A NetworkX graph object with node data.
        tensor (torch.Tensor): An adjacency tensor representation of the graph.
        num_swaps (int): The total number of swap operations planned.
        current_swap (int): The index of the current swap operation.

    Returns:
        tuple: A tuple containing:
            - processed_graph (nx.Graph): The processed graph constructed from nodes and tensor.
            - tensor (torch.Tensor): The input adjacency tensor.
            - mol (rdkit.Chem.Mol): An RDKit molecule object.
            - gemb (torch.Tensor): Global graph embeddings.
            - nemb (torch.Tensor): Node embeddings.
            - distances (torch.Tensor): Pairwise node distances.
            - edge_index (torch.Tensor): Edge indices for PyG.
            - edge_attr (torch.Tensor): Edge attributes for PyG.
            - dosd_positions (torch.Tensor): DOSD distances between nodes.
            - componentes_ant (int): Number of connected components in the graph.
            - fingerprint (torch.Tensor): Morgan fingerprints of the molecule.
    """

    #here I have to use the filling values
    if current_swap >= num_swaps:
        return processed_graph_filling, tensor_filling, mol_filling, gemb_filling, nemb_filling, distances_filling, edge_index_filling, edge_attr_filling, dosd_positions_filling, componentes_ant_filling, fingerprint_filling

    processed_graph = components_to_graph(graph.nodes(data=True), tensor)
    componentes_ant = nx.number_connected_components(processed_graph)
    mol = nx_to_rdkit(processed_graph, False)
    mol.UpdatePropertyCache()
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)
    fingerprint = list(fingerprint)
    fingerprint = torch.Tensor(fingerprint)

    gemb, nemb, distances = embed_graph_nodes_norm(processed_graph)
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(processed_graph)
    dosd_positions = calculate_2d_distances_ordered(processed_graph, list(processed_graph.nodes())) 
    dosd_positions = torch.Tensor(dosd_positions)


    return processed_graph, tensor, mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, componentes_ant, fingerprint    

# filter matrix is used to filter the quadruplets that are not valid
filter_matrix = torch.ones((MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM))
ind = torch.cartesian_prod(torch.arange(MAX_ATOM), torch.arange(MAX_ATOM), torch.arange(MAX_ATOM), torch.arange(MAX_ATOM))
unique_mask = (ind[:, 0] != ind[:, 1]) & \
                            (ind[:, 0] != ind[:, 2]) & \
                            (ind[:, 0] != ind[:, 3]) & \
                            (ind[:, 1] != ind[:, 2]) & \
                            (ind[:, 1] != ind[:, 3]) & \
                            (ind[:, 2] != ind[:, 3])
filter_matrix = unique_mask.reshape((MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM)).float().to(device)

def sample_step_graph(initial_graph, tensor, probs_quadrupletas_mod, all_smiles_molecule, idp, num_swaps, contador_molecula):
    """
    Performs a single sampling step by selecting a quadruplet of nodes and attempting
    to swap edges based on the provided quadruplet probabilities.

    It samples a quadruplet (i1, j1, i2, j2), attempts to remove edges (i1, j1) and
    (i2, j2) and add edges (i1, i2) and (j1, j2). The step is successful if the
    resulting molecule is connected and has not been seen before.

    Args:
        initial_graph (nx.Graph): The starting graph for the step.
        tensor (torch.Tensor): The adjacency tensor of the starting graph.
        probs_quadrupletas_mod (torch.Tensor): A 4D tensor of quadruplet probabilities,
                                               masked and filtered.
        all_smiles_molecule (set): A set containing SMILES strings of previously
                                    encountered molecules to avoid duplicates.
        idp (int): Process ID (used for logging/debugging, currently commented out).
        num_swaps (int): Total number of swaps planned.
        contador_molecula (int): The current molecule/swap counter.

    Returns:
        tuple: A tuple containing:
            - tf (torch.Tensor): The adjacency tensor of the resulting graph after the swap,
                               or the original tensor if the step failed or was skipped.
            - modified_smiles (str or None): The SMILES string of the modified molecule if
                                          the swap was successful, otherwise None.
    """
    
    current_graph_molecule = components_to_graph(initial_graph.nodes(data=True), tensor)
    if contador_molecula >= num_swaps:
        return tensor, None
    current_graph_molecule_copy = current_graph_molecule.copy()

    mask_des = (tensor.to(device) > 0.5).int() 
    mask_haz = (tensor.to(device) < 2.5).int()
    mask_quads = mask_des.unsqueeze(2).unsqueeze(3) * mask_des.unsqueeze(0).unsqueeze(1) * mask_haz.unsqueeze(1).unsqueeze(3) * mask_haz.unsqueeze(0).unsqueeze(2)

    mask_quads = mask_quads

    probs_quadrupletas_mod = probs_quadrupletas_mod * mask_quads

    probs_quadrupletas_mod = probs_quadrupletas_mod * filter_matrix


    flat_prob_tensor = probs_quadrupletas_mod.flatten().double()

    lim_prob = 0.95
    flat_prob_tensor[flat_prob_tensor < lim_prob] = 0


    cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)
    cumulative_distribution2 = cumulative_distribution.clone()
    cumulative_distribution2 /= cumulative_distribution[-1]


    roto = False
    attempts = 0  # Counter for attempts in a single step
    max_attempts = 50 
    rompia = 0
    igual_molecula = 0
    dk = [n for n, d in current_graph_molecule.degree()]
    while not roto and attempts < max_attempts:

            
            shape = probs_quadrupletas_mod.shape
            sampled_position, indice_pos, error = sample_positions(cumulative_distribution2, shape)

            if error or (rompia % 5 == 0 and rompia >= 1) or (igual_molecula % 5 == 0 and igual_molecula >= 1):
                lim_prob -=0.05
                flat_prob_tensor = probs_quadrupletas_mod.flatten().double()

                
                flat_prob_tensor[flat_prob_tensor < lim_prob] = 0
                
                cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)
    
                cumulative_distribution2 = cumulative_distribution.clone()
                cumulative_distribution2 /= cumulative_distribution[-1]
                rompia = 0
                igual_molecula = 0

            else:
                eleccion = sampled_position

                i1 = dk[eleccion[0]]
                j1 = dk[eleccion[1]]
                i2 = dk[eleccion[2]]
                j2 = dk[eleccion[3]]


                if current_graph_molecule.has_edge(i1,j1) == False or current_graph_molecule.has_edge(i2,j2) == False:
                    continue

                # Add the initial SMILES to the set
                current_mol = nx_to_rdkit(current_graph_molecule, False)
                current_smiles = Chem.MolToSmiles(current_mol, allHsExplicit=False)
                all_smiles_molecule.add(current_smiles)
                current_graph_molecule.remove_edge(i1,j1)
                current_graph_molecule.remove_edge(i2,j2)
                current_graph_molecule.add_edge(i1,i2)
                current_graph_molecule.add_edge(j1,j2)

                modified_mol = nx_to_rdkit(current_graph_molecule, False)
                modified_smiles = Chem.MolToSmiles(modified_mol, allHsExplicit=False)
                componentes_act = nx.number_connected_components(current_graph_molecule)
                if (componentes_act <= 1) and (modified_smiles not in all_smiles_molecule):
                    rompia = 0 
                    igual_molecula = 0
                    roto = True
                    
                    final_graph = current_graph_molecule.copy()
                   
                else:
                    if (componentes_act > 1):
                        rompia += 1 
                    else:
                        igual_molecula += 1
                    current_graph_molecule = current_graph_molecule_copy.copy()
                    attempts += 1
    if attempts >= max_attempts:
        tf, _, _ = embed_edges_manuel(current_graph_molecule, list(current_graph_molecule.nodes()))
        return tf, None, 

    tf, _, _ = embed_edges_manuel(final_graph, list(final_graph.nodes()))
    return tf, modified_smiles
