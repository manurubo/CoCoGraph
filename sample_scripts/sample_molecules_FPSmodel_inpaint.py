"""
Molecular Inpainting with CoCoGraph

This script implements molecular inpainting by:
1. Taking an original molecule (e.g., paracetamol)
2. Generating a random fragment from a specified molecular formula
3. Attaching the fragment to the original molecule at a random attachment point
   - Both attachment points must have at least one H atom
   - The two H atoms are removed and replaced with a bond between the molecules
4. Running the diffusion model to refine only the fragment region
   - A mask protects the original molecule's internal heavy atom bonds
   - The model can modify: fragment bonds, fragment-to-original connections, and original H positions

Two modes of operation:
A) Random sampling: Samples different molecules from a dataset for each batch
B) Target molecule: Uses the same specified molecule for all inpainting operations

Usage examples:

1. Random sampling mode (explore diverse molecules):
    python sample_molecules_FPSmodel_inpaint.py \
        --fragment_formula C3H6 \
        --output_dir_suffix Inpaint_C3H6_Random \
        --batch_size_process 10 \
        --batch_size_sample 100

2. Target molecule mode (focus on one molecule, e.g., paracetamol):
    python sample_molecules_FPSmodel_inpaint.py \
        --fragment_formula C2H4 \
        --target_smiles "CC(=O)Nc1ccc(O)cc1" \
        --output_dir_suffix Inpaint_Paracetamol_C2H4 \
        --batch_size_process 10 \
        --batch_size_sample 100

Key parameters:
    --fragment_formula: Molecular formula of the fragment to attach (e.g., C2H4, CH2O, C3H6)
    --target_smiles: Optional SMILES string of specific molecule to use (if not provided, samples randomly)
    --output_dir_suffix: Name for the output directory
    --batch_size_process: Number of molecules to process in parallel
    --batch_size_sample: Number of molecules to sample per batch
"""

import os
import sys
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib_functions.sample_utils import calculate_data_molecule_fps, sample_step_graph
from lib_functions.libraries import *
from lib_functions.config import *
from lib_functions.models import GINEdgeQuadrupletPredictor_MorganFP,  GINETimePredictor_MorganFP
from lib_functions.data_preparation_utils import embed_edges_manuel
from lib_functions.adjacency_utils import components_to_graph
from lib_functions.data_loader import build_dataset_alejandro
from lib_functions.data_loader import build_dataset_from_formulas
from lib_functions.adjacency_utils import nx_to_rdkit
from lib_functions.adjacency_utils import connected_double_edge_swap
from lib_functions.formula_utils import build_gt_from_formula
from lib_functions.data_preparation_utils import smiles_to_graph

from rdkit import Chem
from copy import deepcopy
import gc
import argparse
import random 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import json

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_atoms_with_hydrogens(graph):
    """
    Find all heavy atoms in the graph that have at least one hydrogen neighbor.
    
    Args:
        graph (nx.MultiGraph): The molecular graph
    
    Returns:
        list: List of heavy atom node names that have H connections
    """
    atoms_with_h = []
    for node, data in graph.nodes(data=True):
        if data['label'] != 'H':  # Heavy atom
            # Check if any neighbor is hydrogen
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor]['label'] == 'H':
                    atoms_with_h.append(node)
                    break
    return atoms_with_h


def attach_fragment_to_molecule(original_graph, fragment_graph, rng=None):
    """
    Attaches a fragment graph to an original molecule graph at random attachment points.
    Removes one hydrogen from each and creates a bond between them.
    
    Args:
        original_graph (nx.MultiGraph): The original molecule graph
        fragment_graph (nx.MultiGraph): The fragment graph to attach
        rng: Random number generator (optional)
    
    Returns:
        tuple: (composite_graph, original_nodes, fragment_nodes, attachment_bond)
            - composite_graph: The merged graph
            - original_nodes: List of node names from original molecule
            - fragment_nodes: List of node names from fragment
            - attachment_bond: Tuple (original_atom, fragment_atom) of the attachment bond
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Find atoms with hydrogens in both graphs
    original_atoms_with_h = get_atoms_with_hydrogens(original_graph)
    fragment_atoms_with_h = get_atoms_with_hydrogens(fragment_graph)
    
    if not original_atoms_with_h:
        raise ValueError("Original molecule has no atoms with hydrogens for attachment")
    if not fragment_atoms_with_h:
        raise ValueError("Fragment has no atoms with hydrogens for attachment")
    
    # Randomly select attachment points
    original_attach = rng.choice(original_atoms_with_h)
    fragment_attach = rng.choice(fragment_atoms_with_h)
    
    # Create composite graph starting with original
    composite = original_graph.copy()
    
    # Keep track of original nodes
    original_nodes = list(original_graph.nodes())
    
    # Rename fragment nodes to avoid collisions and add to composite
    fragment_node_mapping = {}
    fragment_nodes = []
    for node, data in fragment_graph.nodes(data=True):
        # Create unique name by prefixing with 'frag_'
        new_name = f"frag_{node}"
        fragment_node_mapping[node] = new_name
        composite.add_node(new_name, **data)
        fragment_nodes.append(new_name)
    
    # Add fragment edges with renamed nodes
    for u, v, data in fragment_graph.edges(data=True):
        composite.add_edge(fragment_node_mapping[u], fragment_node_mapping[v], **data)
    
    # Find and remove one hydrogen from original attach point
    h_to_remove_original = None
    for neighbor in original_graph.neighbors(original_attach):
        if original_graph.nodes[neighbor]['label'] == 'H':
            h_to_remove_original = neighbor
            break
    
    # Find and remove one hydrogen from fragment attach point
    fragment_attach_renamed = fragment_node_mapping[fragment_attach]
    h_to_remove_fragment = None
    for neighbor in list(composite.neighbors(fragment_attach_renamed)):
        if composite.nodes[neighbor]['label'] == 'H' and neighbor in fragment_nodes:
            h_to_remove_fragment = neighbor
            break
    
    # Remove the hydrogens
    if h_to_remove_original:
        composite.remove_node(h_to_remove_original)
        original_nodes.remove(h_to_remove_original)
    if h_to_remove_fragment:
        composite.remove_node(h_to_remove_fragment)
        fragment_nodes.remove(h_to_remove_fragment)
    
    # Create the attachment bond
    composite.add_edge(original_attach, fragment_attach_renamed)
    
    return composite, original_nodes, fragment_nodes, (original_attach, fragment_attach_renamed)


def create_inpainting_mask(composite_graph, original_nodes, fragment_nodes, node_list, padded_size):
    """
    Creates a mask for inpainting that protects internal bonds of the original molecule.
    
    The mask allows changes to:
    - Bonds within the fragment
    - Bonds between original and fragment (including original H atoms)
    - Original heavy atom to H bonds (but only for connections to fragment)
    
    The mask protects:
    - Bonds between original heavy atoms
    
    Args:
        composite_graph (nx.MultiGraph): The composite molecule graph
        original_nodes (list): Node names from the original molecule
        fragment_nodes (list): Node names from the fragment
        node_list (list): Ordered list of all nodes (for indexing)
        padded_size (int): The padded size to match the adjacency tensor
    
    Returns:
        torch.Tensor: A binary mask of shape (padded_size, padded_size) where 1 means editable, 0 means protected
    """
    # Create mask with padded size
    mask = torch.ones((padded_size, padded_size), dtype=torch.float32)
    
    # Create index mapping
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Identify heavy atoms in original molecule
    original_heavy_atoms = []
    for node in original_nodes:
        if node in node_to_idx and composite_graph.nodes[node]['label'] != 'H':
            original_heavy_atoms.append(node)
    
    # Protect bonds between original heavy atoms (set to 0)
    for i, node_i in enumerate(original_heavy_atoms):
        idx_i = node_to_idx[node_i]
        for j, node_j in enumerate(original_heavy_atoms):
            if i != j:
                idx_j = node_to_idx[node_j]
                mask[idx_i, idx_j] = 0
                mask[idx_j, idx_i] = 0
    
    return mask


def sample_step_graph_inpaint(initial_graph, tensor, probs_quadrupletas_mod, all_smiles_molecule, idp, num_swaps, contador_molecula, inpainting_mask):
    """
    Modified version of sample_step_graph that respects an inpainting mask.
    The inpainting mask protects certain edges from being modified.
    
    Args:
        Same as sample_step_graph, plus:
        inpainting_mask (torch.Tensor): Binary mask (N, N) where 0 means protected, 1 means editable
    
    Returns:
        tuple: (tensor, smiles) same as sample_step_graph
    """
    from lib_functions.sample_utils import sample_positions, filter_matrix
    
    current_graph_molecule = components_to_graph(initial_graph.nodes(data=True), tensor)
    if contador_molecula >= num_swaps:
        return tensor, None
    current_graph_molecule_copy = current_graph_molecule.copy()

    mask_des = (tensor.to(device) > 0.5).int() 
    mask_haz = (tensor.to(device) < 2.5).int()
    mask_quads = mask_des.unsqueeze(2).unsqueeze(3) * mask_des.unsqueeze(0).unsqueeze(1) * mask_haz.unsqueeze(1).unsqueeze(3) * mask_haz.unsqueeze(0).unsqueeze(2)

    # Apply inpainting mask: convert to 4D mask for quadruplets
    # A quadruplet (i,j,i2,j2) is only valid if we can modify edges (i,j) and (i2,j2)
    inpainting_mask_4d = inpainting_mask.unsqueeze(2).unsqueeze(3) * inpainting_mask.unsqueeze(0).unsqueeze(1)
    inpainting_mask_4d = inpainting_mask_4d.to(device)
    
    probs_quadrupletas_mod = probs_quadrupletas_mod * mask_quads * inpainting_mask_4d
    probs_quadrupletas_mod = probs_quadrupletas_mod * filter_matrix

    flat_prob_tensor = probs_quadrupletas_mod.flatten().double()

    lim_prob = 0.95
    flat_prob_tensor[flat_prob_tensor < lim_prob] = 0

    cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)
    
    if cumulative_distribution[-1] == 0:
        return tensor, None
    
    cumulative_distribution = cumulative_distribution / cumulative_distribution[-1]

    count = 0
    tf = tensor
    modified_smiles = None
    while count < 500:
        position_4d, index, error = sample_positions(cumulative_distribution, probs_quadrupletas_mod.shape)
        
        if error:
            count += 1
            continue
            
        i1, j1, i2, j2 = position_4d[0].item(), position_4d[1].item(), position_4d[2].item(), position_4d[3].item()
        
        # Verify the selected edges are not protected by the inpainting mask
        if inpainting_mask[i1, j1] == 0 or inpainting_mask[i2, j2] == 0:
            count += 1
            continue
        
        # Attempt the swap
        try:
            current_graph_molecule_copy.remove_edge(list(current_graph_molecule_copy.nodes())[i1], list(current_graph_molecule_copy.nodes())[j1])
            current_graph_molecule_copy.remove_edge(list(current_graph_molecule_copy.nodes())[i2], list(current_graph_molecule_copy.nodes())[j2])
            current_graph_molecule_copy.add_edge(list(current_graph_molecule_copy.nodes())[i1], list(current_graph_molecule_copy.nodes())[i2])
            current_graph_molecule_copy.add_edge(list(current_graph_molecule_copy.nodes())[j1], list(current_graph_molecule_copy.nodes())[j2])
        except:
            count += 1
            current_graph_molecule_copy = current_graph_molecule.copy()
            continue

        # Check connectivity
        if not nx.is_connected(current_graph_molecule_copy):
            count += 1
            current_graph_molecule_copy = current_graph_molecule.copy()
            continue
        
        # Check for duplicate SMILES
        try:
            mol_temporal = nx_to_rdkit(current_graph_molecule_copy, False)
            smiles_temporal = Chem.MolToSmiles(mol_temporal)
            
            if smiles_temporal in all_smiles_molecule:
                count += 1
                current_graph_molecule_copy = current_graph_molecule.copy()
                continue
            
            # Success!
            tf, _, _ = embed_edges_manuel(current_graph_molecule_copy, list(current_graph_molecule_copy.nodes()))
            modified_smiles = smiles_temporal
            break
        except:
            count += 1
            current_graph_molecule_copy = current_graph_molecule.copy()
            continue
    
    return tf, modified_smiles


def process_batch(conjunto, model, num, b_molecule, cantidad, time_model, fragment_formula='C2H4', fragment_library=None):

    # Move the model to the device
    model = model.to(device)
    time_model = time_model.to(device) 

    # Initialize the dataframe
    df_generated_temporal = pd.DataFrame()

    # Get the valid graph and edge
    valid_graph_b, valid_edge_b, smiles, _ = conjunto

    # Load valence/charge/radical configurations for fragment generation
    valid_valences_json_path = os.path.join('Data', 'valid_valences.json')
    charge_weights_json_path = os.path.join('Data', 'charge_symbol_weights.json')
    radical_weights_json_path = os.path.join('Data', 'radical_symbol_weights.json')

    valid_valences = None
    valence_weights = None
    charge_symbol_weights = None
    radical_symbol_weights = None

    if os.path.exists(valid_valences_json_path):
        try:
            with open(valid_valences_json_path, 'r') as f:
                raw = json.load(f)
            valid_valences = {}
            valence_weights = {}
            for k, vals in raw.items():
                sym, chs = k.split('__')
                if isinstance(vals, list):
                    valid_valences[(sym, int(chs))] = set(int(v) for v in vals)
                elif isinstance(vals, dict):
                    valid_valences[(sym, int(chs))] = set(int(v) for v in vals.keys())
                    valence_weights[(sym, int(chs))] = {int(v): float(p) for v, p in vals.items()}
                else:
                    valid_valences[(sym, int(chs))] = set()
        except Exception as e:
            print(f"Warning: Could not load valence weights: {e}")
            valid_valences = None
            valence_weights = None

    if os.path.exists(charge_weights_json_path):
        try:
            with open(charge_weights_json_path, 'r') as cf:
                charge_symbol_weights = json.load(cf)
        except Exception:
            charge_symbol_weights = None

    if os.path.exists(radical_weights_json_path):
        try:
            with open(radical_weights_json_path, 'r') as rf:
                radical_symbol_weights = json.load(rf)
        except Exception:
            radical_symbol_weights = None

    # Initialize lists for composite molecules
    composite_graph_list = []
    composite_edge_list = []
    original_smiles_list = []
    fragment_formulas_used = []  # Track which fragment was used for each molecule
    inpainting_masks = []
    num_swaps = []
    count = 0
    error_count = 0

    # Iterate over each molecule in the batch
    for cuenta_smiles in range(valid_edge_b.size(0)):
        try:
            # Get the original molecule graph
            original_graph = valid_graph_b[count]
            original_smi = smiles[count]
            
            # Select fragment formula (from library or fixed)
            if fragment_library is not None and len(fragment_library) > 0:
                # Randomly sample from fragment library
                current_fragment_formula = np.random.choice(fragment_library)
            else:
                # Use fixed formula
                current_fragment_formula = fragment_formula
            
            # Generate a random fragment from the selected formula
            try:
                fragment_graph = build_gt_from_formula(
                    current_fragment_formula,
                    randomize_swaps=0,
                    valence_weights=valence_weights,
                    charge_symbol_weights=charge_symbol_weights,
                    max_sampling_retries=100,
                    allow_radicals=(radical_symbol_weights is not None),
                    radical_weights=radical_symbol_weights,
                )
            except Exception as e:
                print(f"Error generating fragment from formula '{current_fragment_formula}': {e}")
                error_count += 1
                count += 1
                continue
            
            # Attach the fragment to the original molecule
            try:
                composite_graph, original_nodes, fragment_nodes, attachment_bond = attach_fragment_to_molecule(
                    original_graph, fragment_graph
                )
            except Exception as e:
                print(f"Error attaching fragment to molecule {count}: {e}")
                error_count += 1
                count += 1
                continue
            
            # Embed the edges of the composite graph
            try:
                composite_tensor, _, _ = embed_edges_manuel(composite_graph, list(composite_graph.nodes()))
            except Exception as e:
                print(f"Error embedding composite graph edges: {e}")
                error_count += 1
                count += 1
                continue
            
            # Create inpainting mask with the same padded size as the tensor
            node_list = list(composite_graph.nodes())
            padded_size = composite_tensor.shape[0]  # Get the padded size from the tensor
            inpainting_mask = create_inpainting_mask(composite_graph, original_nodes, fragment_nodes, node_list, padded_size)
            
            # Calculate number of swaps based on fragment size
            # We'll use sigma * fragment_edges as the number of steps
            sigma_i = 0.5
            fragment_edges = fragment_graph.number_of_edges()
            num_cambios = max(1, math.ceil(sigma_i * fragment_edges))
            
            # Store the composite information (including the fragment formula used)
            composite_graph_list.append(composite_graph)
            composite_edge_list.append(composite_tensor.clone())
            original_smiles_list.append(original_smi)
            fragment_formulas_used.append(current_fragment_formula)  # Track which fragment was used
            inpainting_masks.append(inpainting_mask)
            num_swaps.append(num_cambios)
            
            count += 1
        except Exception as e:
            print(f"Unexpected error processing molecule {count}: {e}")
            error_count += 1
            count += 1
            continue
    
    if error_count > 0:
        print(f"Warning: {error_count} molecules skipped due to errors")
    
    if len(composite_graph_list) == 0:
        print("Error: No valid composite molecules generated")
        return df_generated_temporal

    # Initialize the variables
    molecule_counter = 0 # counter for the molecules
    num_swaps_max = max(num_swaps) # maximum number of swaps for the current batch
    all_smiles_for_all_molecules = [] # this is a list of sets of smiles for each molecule
    best_time_all_molecules = [0.5] * len(num_swaps) # should initialize with all 0.5s
    best_tensor_all_molecules = [None] * len(num_swaps) # should initialize with all None

    # Iterate over the molecules for the maximum number of swaps, if a molecule has less swaps we have a default molecule and we dont take it into account
    while molecule_counter < num_swaps_max:

        print(f"Processing molecules step {molecule_counter} of {num_swaps_max}")

        # if the molecule counter is 0, we initialize the time predictions and the tensors
        if molecule_counter == 0:
            time_predictions = [0.5] * len(num_swaps)
            tensors_allmolecules = composite_edge_list  # Use composite graphs instead of noisy graphs
            # put a open set for each molecule
            for i in range(len(num_swaps)):
                all_smiles_for_all_molecules.append(set())
        
        # Calculate the data for each molecule at current step
        futures = [executor_gpu.submit(calculate_data_molecule_fps, graph, tensor, num_swap, molecule_counter) for graph, tensor, num_swap in zip(composite_graph_list, tensors_allmolecules, num_swaps)]

        # Get the results
        results = [future.result() for future in futures]

        # Initialize the dataset
        dataset = []

        # Iterate over the results and get the data
        for result, prediction_time in zip(results, time_predictions):
            processed_graph, tensor, mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, componentes_ant, fingerprint = result

            d = Data(
                x=nemb,
                edge_index=edge_index,
                y=tensor,
                xA=gemb,
                edge_attr=edge_attr,
                noiselevel=torch.tensor(prediction_time, device=device),
                distances=torch.Tensor(distances),
                dosd_distances=dosd_positions,
                morgan_fp=fingerprint
            ).to(device)
            dataset.append(d)

        # Diffusion model prediction for each molecule
        probs_quadrupletas_mod_list = []
        for d in dataset:
            _,_, probs_quadrupletas_mod = model(d)
            probs_quadrupletas_mod_list.append(probs_quadrupletas_mod)


        # Time model prediction for each molecule
        time_predictions = []
        for d in dataset:
            time_pred = time_model(d)
            time_predictions.append(time_pred.detach().cpu().item())

        # Update the previous best time and tensor for each molecule
        for i in range(len(num_swaps)):
            if molecule_counter == 0 or (molecule_counter < num_swaps[i] and time_predictions[i] < best_time_all_molecules[i]): 
                best_time_all_molecules[i] = time_predictions[i]
                best_tensor_all_molecules[i] = tensors_allmolecules[i]

        molecule_counter += 1 # next step
        
        # sample the next step graph for each molecule using inpainting masks
        results = []
        futures = [executor_gpu.submit(sample_step_graph_inpaint, graph, tensor.clone(), probs_quadrupletas.detach(), all_smiles_molecule, idp, num_swaps[idp], molecule_counter-1, mask) for graph, tensor, probs_quadrupletas, all_smiles_molecule, idp, mask in zip(composite_graph_list, tensors_allmolecules, probs_quadrupletas_mod_list, all_smiles_for_all_molecules, range(len(num_swaps)), inpainting_masks)]
        
        # Get the results
        results = [future.result() for future in futures]

        # Iterate over the results
        for index, result in enumerate(results):
            tensor, smiles_result = result
            if smiles_result is not None: # if smiles is not None, add the smiles to the set, if is None, the molecule has been generated
                all_smiles_for_all_molecules[index].add(smiles_result)
                tensors_allmolecules[index] = tensor
            else:
                continue
                
    # After the loop, find the molecule with the smallest time prediction, excluding the original molecule
    for i in range(len(num_swaps)):
        # get the best molecule for each molecule
        best_tensor = best_tensor_all_molecules[i]

        # Use the best molecule for further processing
        g_gen = components_to_graph(composite_graph_list[i].nodes(data=True), best_tensor)

        # Convert the molecule to RDKit format without hydrogens
        mol_des = nx_to_rdkit(g_gen, False)

        # Add the SMILES and descriptors to the DataFrame
        smiles_str = Chem.MolToSmiles(mol_des)
        original_smi = original_smiles_list[i]
        fragment_used = fragment_formulas_used[i] if i < len(fragment_formulas_used) else fragment_formula

        # append the smiles to the dataframe (including original SMILES for comparison)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_des)
        num_atoms = g_gen.number_of_nodes()
        df_generated_temporal = df_generated_temporal._append({
            'smiles': smiles_str, 
            'original_smiles': original_smi,
            'molecular_formula': formula, 
            'num_atoms': num_atoms,
            'fragment_formula': fragment_used  # Save the actual fragment used (not the default parameter)
        }, ignore_index=True)

    return df_generated_temporal
    


if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Molecular inpainting script using CoCoGraph.')
    parser.add_argument('--input_smiles_csv', type=str, default='Data/molecules_lt70atoms_annotated.csv', help='Path to the input CSV file containing SMILES.')
    parser.add_argument('--output_dir_suffix', type=str, default='Inpaint_Test', help='Suffix for the output directory name.')
    parser.add_argument('--fragment_formula', type=str, default='C2H4', help='Molecular formula of the fragment to attach (e.g., C2H4, CH2O, C3H6). Ignored if --fragment_library is provided.')
    parser.add_argument('--fragment_library', type=str, default='', help='Path to file with fragment formulas (one per line). If provided, randomly samples from this library instead of using fixed formula.')
    parser.add_argument('--target_smiles', type=str, default='', help='Specific SMILES to use as base molecule (e.g., paracetamol: CC(=O)Nc1ccc(O)cc1). If not provided, samples randomly from input CSV.')
    parser.add_argument('--model_checkpoint_path', type=str, default='models/FPS_diffusion/model_epoch_1_slice_22.pth', help='Path to the main model checkpoint.')
    parser.add_argument('--time_model_checkpoint_path', type=str, default='models/FPS_time/model_epoch_2_slice_22.pth', help='Path to the time prediction model checkpoint.')
    parser.add_argument('--batch_size_sample', type=int, default=100, help='Number of SMILES to sample from the input file in each main loop iteration.')
    parser.add_argument('--batch_size_process', type=int, default=50, help='Batch size for processing molecules within process_batch.')
    parser.add_argument('--save_every_n_batches', type=int, default=50, help='Save cumulative results every N sampling batches.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel execution.')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Determine if using target SMILES or random sampling
    use_target_smiles = bool(args.target_smiles)
    
    # Load fragment library if provided
    use_fragment_library = bool(args.fragment_library)
    fragment_library = []
    
    if use_fragment_library:
        print(f"Loading fragment library from {args.fragment_library}...")
        try:
            with open(args.fragment_library, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        fragment_library.append(line)
            print(f"Loaded {len(fragment_library)} fragments from library")
        except FileNotFoundError:
            print(f"Error: Fragment library file not found: {args.fragment_library}")
            exit(1)
        except Exception as e:
            print(f"Error loading fragment library: {e}")
            exit(1)
    
    print(f"\n=== Molecular Inpainting Configuration ===")
    if use_fragment_library:
        print(f"Fragment mode: Random from library ({len(fragment_library)} fragments)")
        print(f"Library file: {args.fragment_library}")
        print(f"Sample fragments: {', '.join(fragment_library[:5])}{'...' if len(fragment_library) > 5 else ''}")
    else:
        print(f"Fragment mode: Fixed formula")
        print(f"Fragment formula: {args.fragment_formula}")
    print(f"Output directory: {args.output_dir_suffix}")
    if use_target_smiles:
        print(f"Target molecule: {args.target_smiles}")
        print(f"Molecule mode: Single molecule repeated")
    else:
        print(f"Molecule mode: Random sampling from {args.input_smiles_csv}")
    print(f"Batch size (sample): {args.batch_size_sample}")
    print(f"Batch size (process): {args.batch_size_process}")
    print(f"==========================================\n")

    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # Read input SMILES or use target SMILES
    if use_target_smiles:
        # Validate the target SMILES
        test_mol = Chem.MolFromSmiles(args.target_smiles)
        if test_mol is None:
            print(f"Error: Invalid target SMILES: {args.target_smiles}")
            exit(1)
        # Create a list with just the target SMILES repeated
        selected_smiles = [args.target_smiles]
        print(f"Using target molecule: {args.target_smiles}")
        mol_name = Chem.MolToSmiles(test_mol)  # Canonical SMILES
        print(f"Canonical form: {mol_name}")
    else:
        # Random sampling mode
        df = pd.read_csv(args.input_smiles_csv)
        df = df.sample(frac=1, random_state=1111).reset_index(drop=True)
        selected_smiles = df['smiles'].tolist()
        print(f"Loaded {len(selected_smiles)} molecules from {args.input_smiles_csv}")

    # set the date for the molecule generation
    # current_date = datetime.now().strftime("%Y%m%d") # Use suffix instead
    str_date = args.output_dir_suffix

    # Create directories if they don't exist
    molsgen_dir = os.path.join("mols_gen", str_date)
    molsgen_dir_temporal = os.path.join("mols_gen", str_date, "temporal_dfs")

    # create the directories
    os.makedirs(molsgen_dir, exist_ok=True)
    os.makedirs(molsgen_dir_temporal, exist_ok=True)

    # load the models
    model = GINEdgeQuadrupletPredictor_MorganFP()
    time_model = GINETimePredictor_MorganFP()

    # try to load the models
    try:
        checkpoint = torch.load(args.model_checkpoint_path)
        checkpoint_time = torch.load(args.time_model_checkpoint_path)
    except FileNotFoundError:
        print(f"Error: Could not find model checkpoints at specified paths:")
        print(f"  Main model: {args.model_checkpoint_path}")
        print(f"  Time model: {args.time_model_checkpoint_path}")
        exit()
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        exit()

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        # epoch = checkpoint['epoch'] # epoch might not be needed if just evaluating
    else:
        print(f"Warning: Checkpoint for main model at {args.model_checkpoint_path} loaded as None.")


    if checkpoint_time is not None:
        time_model.load_state_dict(checkpoint_time['model_state_dict'])
        # epoch = checkpoint['epoch'] # epoch might not be needed if just evaluating
    else:
        print(f"Warning: Checkpoint for time model at {args.time_model_checkpoint_path} loaded as None.")

    # set the models to evaluation mode
    model.eval()
    time_model.eval()
    

    # Process smiles in batches
    batch_size = args.batch_size_sample # Renamed for clarity internally
    quantity = args.batch_size_process # Use argument value
    batches_to_store = args.save_every_n_batches # Use argument value
    total_batches_processed = 0
    df_generated_total = pd.DataFrame() # dataframe to store all generated molecules
    
    print(f"Starting inpainting loop (fragment: {args.fragment_formula})...")
    
    while True:  # Run indefinitely
        # Sample SMILES based on mode
        if use_target_smiles:
            # Use the same target SMILES repeated batch_size times
            batch_smiles = [args.target_smiles] * batch_size
            print(f"Processing batch {total_batches_processed} with target molecule repeated {batch_size} times")
        else:
            # Randomly sample batch_size SMILES with replacement to generate every time a different set of molecules
            batch_smiles = np.random.choice(selected_smiles, size=batch_size, replace=True)

        # Prepare the geometric pytorch data loaders for this batch
        train_dl, validation_dl, test_dl, bonds_perc = build_dataset_alejandro(
            batch_smiles,
            ftr=1.0, fva=0.0, # all molecules are used for training data as we dont care
            bs=quantity, # Use argument value
            min_atom=5,
        )

        # initialize the dataframes
        df_generated_batch_slice = pd.DataFrame() # dataframe to store the generated molecules for each batch
        df_generated_batch = pd.DataFrame() # dataframe to store all generated molecules
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor_gpu: # Use argument value
            
            # Map process_batch function over all batches   
            futures = []
            for num, conjunto in enumerate(train_dl):
                df_generated_batch_slice = process_batch(
                    conjunto, model, num, total_batches_processed, quantity, time_model, 
                    fragment_formula=args.fragment_formula,
                    fragment_library=fragment_library if use_fragment_library else None
                )
                df_generated_batch_slice.to_csv(f"mols_gen/{str_date}/batch_{total_batches_processed}_{num}_generated_molecules.csv", index=False)
                df_generated_batch = df_generated_batch._append(df_generated_batch_slice, ignore_index=True)

        # Save intermediate results for the whole sampling batch
        df_generated_batch.to_csv(f"{molsgen_dir}/generated_molecules_batch_{total_batches_processed}.csv", index=False) # save the generated molecules for each sampling batch

        df_generated_total = df_generated_total._append(df_generated_batch, ignore_index=True) # append the generated molecules to the total dataframe
        
        # Clear memory  
        del train_dl, validation_dl, test_dl, df_generated_batch, df_generated_batch_slice, futures, batch_smiles
        gc.collect()

        total_batches_processed += 1

        # Save total results every batches_to_store batches 
        if total_batches_processed % batches_to_store == 0:
            current_molecules = total_batches_processed * batch_size
            df_generated_total.to_csv(f"{molsgen_dir}/all_generated_molecules.csv", index=False)
            print(f"Saved cumulative results for {current_molecules} molecules after {total_batches_processed} batches.")
