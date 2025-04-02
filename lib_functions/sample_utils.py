from lib_functions.adjacency_utils import components_to_graph, nx_to_rdkit
from lib_functions.config import *
from lib_functions.libraries import *
import torch
import itertools

from lib_functions.data_preparation_utils import embed_edges_manuel, embed_edges_with_cycle_sizes_norm, smiles_to_graph, embed_graph_nodes_norm, calculate_2d_distances_ordered
from rdkit.Chem import AllChem




def sample_positions(cumulative_distribution, shape):
    # Generar un valor aleatorio entre 0 y 1
    random_value = torch.rand(1).item()
    
    # Encontrar el índice correspondiente en la distribución acumulativa
    index = torch.searchsorted(cumulative_distribution, random_value).item()
    # Convertir el índice 1D a índices 4D
    position_4d = torch.unravel_index(torch.tensor(index), shape)

    error = False
    if index == 24010000:
        error = True
    
    return position_4d, index, error


# filling molecule: 'CN(CC(=O)O)C(=N)N ' Creatine hehe
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

    #print(smiles)
    #print(graph)
    #print(graph.nodes())
    #here I have to use the filling values
    if current_swap >= num_swaps:
        return processed_graph_filling, tensor_filling, mol_filling, gemb_filling, nemb_filling, distances_filling, edge_index_filling, edge_attr_filling, dosd_positions_filling, componentes_ant_filling

    #start = time.time()
    processed_graph = components_to_graph(graph.nodes(data=True), tensor)
    componentes_ant = nx.number_connected_components(processed_graph)
    mol = nx_to_rdkit(processed_graph, True)
    #print(processed_graph.nodes())
    #print("Nodes and their neighbors:")
    #for node in processed_graph.nodes():
    #    neighbors = list(processed_graph.neighbors(node))
        #print(f"Node {node} has neighbors: {neighbors}")
    #img = Draw.MolToImage(mol)
    #img.save(f'mols_gen_prueba/molecule_sampling.png')
    #mol.UpdatePropertyCache()
    #end = time.time()
    #print(f"Create processed graph data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)

    #start = time.time()
    gemb, nemb, distances = embed_graph_nodes_norm(processed_graph)
    #end = time.time()
    #print(f"Embed graph nodes data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)
    #start = time.time() 
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(processed_graph)
    #end = time.time()
    #print(f"Embed edges with cycle sizes data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)
    #start = time.time() 
    dosd_positions = calculate_2d_distances_ordered(processed_graph, list(processed_graph.nodes())) 
    dosd_positions = torch.Tensor(dosd_positions)
    #end = time.time()
    #print(f"Calculate 2d distances data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)

    return processed_graph, tensor, mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, componentes_ant    

def calculate_data_molecule_fps(graph, tensor, num_swaps, current_swap):

    #print(graph)
    #print(graph.nodes())
    #here I have to use the filling values
    if current_swap >= num_swaps:
        return processed_graph_filling, tensor_filling, mol_filling, gemb_filling, nemb_filling, distances_filling, edge_index_filling, edge_attr_filling, dosd_positions_filling, componentes_ant_filling, fingerprint_filling

    #start = time.time()
    processed_graph = components_to_graph(graph.nodes(data=True), tensor)
    componentes_ant = nx.number_connected_components(processed_graph)
    mol = nx_to_rdkit(processed_graph, False)
    mol.UpdatePropertyCache()
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3)
    fingerprint = list(fingerprint)
    fingerprint = torch.Tensor(fingerprint)

    #start = time.time()
    gemb, nemb, distances = embed_graph_nodes_norm(processed_graph)
    #end = time.time()
    #print(f"Embed graph nodes data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)
    #start = time.time() 
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(processed_graph)
    #end = time.time()
    #print(f"Embed edges with cycle sizes data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)
    #start = time.time() 
    dosd_positions = calculate_2d_distances_ordered(processed_graph, list(processed_graph.nodes())) 
    dosd_positions = torch.Tensor(dosd_positions)
    #end = time.time()
    #print(f"Calculate 2d distances data for molecules step {current_swap} of {num_swaps} took: {end - start}", flush=True)

    return processed_graph, tensor, mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, componentes_ant, fingerprint    

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
    
    current_graph_molecule = components_to_graph(initial_graph.nodes(data=True), tensor)
    if contador_molecula >= num_swaps:
        return tensor, None
    #start = time.time()
    current_graph_molecule_copy = current_graph_molecule.copy()
    #end = time.time()
    #print(f"P: {idp},Copy graph took: {end - start}", flush=True)

    #start = time.time()
    mask_des = (tensor.to(device) > 0.5).int() 
    mask_haz = (tensor.to(device) < 2.5).int()
    mask_quads = mask_des.unsqueeze(2).unsqueeze(3) * mask_des.unsqueeze(0).unsqueeze(1) * mask_haz.unsqueeze(1).unsqueeze(3) * mask_haz.unsqueeze(0).unsqueeze(2)

    mask_quads = mask_quads
    #end = time.time()
    #print(f"P: {idp},Mask quads took: {end - start}", flush=True)

    #start = time.time()
    probs_quadrupletas_mod = probs_quadrupletas_mod * mask_quads
    #end = time.time()
    #print(f"P: {idp},Probs quads mod took: {end - start}", flush=True)

    #start = time.time()
    probs_quadrupletas_mod = probs_quadrupletas_mod * filter_matrix
    #end = time.time()
    #print(f"P: {idp},Filter matrix took: {end - start}", flush=True)

    #start = time.time()
    flat_prob_tensor = probs_quadrupletas_mod.flatten().double()
    #end = time.time()
    #print(f"P: {idp},Flatten took: {end - start}", flush=True)

    #start = time.time()
    lim_prob = 0.95
    flat_prob_tensor[flat_prob_tensor < lim_prob] = 0
    #end = time.time()
    #print(f"P: {idp},Lim prob took: {end - start}", flush=True)

    #start = time.time()
    cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)
    cumulative_distribution2 = cumulative_distribution.clone()
    cumulative_distribution2 /= cumulative_distribution[-1]
    #end = time.time()
    #print(f"P: {idp},Cumulative distribution took: {end - start}", flush=True)


    roto = False
    attempts = 0  # Counter for attempts in a single step
    max_attempts = 50 # si lo dejo asi da resultados igual de buenos?
    rompia = 0
    igual_molecula = 0
    #start = time.time()
    dk = [n for n, d in current_graph_molecule.degree()]
    while not roto and attempts < max_attempts:
            #print(attempts, flush=True)

            # 3. Utilizar torch.multinomial para muestrear una cuádrupleta basada en la distribución de probabilidad
            shape = probs_quadrupletas_mod.shape
            sampled_position, indice_pos, error = sample_positions(cumulative_distribution2, shape)
             # Add the initial SMILES to the set
            
            #print(error)

            if error or (rompia % 5 == 0 and rompia >= 1) or (igual_molecula % 5 == 0 and igual_molecula >= 1):
                #print(lim_prob)
                lim_prob -=0.05
                flat_prob_tensor = probs_quadrupletas_mod.flatten().double()

                # print the top 5 maximum values of flat_prob_tensor
                #print(flat_prob_tensor.topk(5))
    
                
                flat_prob_tensor[flat_prob_tensor < lim_prob] = 0
                
                cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)
    
                cumulative_distribution2 = cumulative_distribution.clone()
                cumulative_distribution2 /= cumulative_distribution[-1]
                rompia = 0
                igual_molecula = 0
                #attempts += 1

            else:
                eleccion = sampled_position

                i1 = dk[eleccion[0]]
                j1 = dk[eleccion[1]]
                i2 = dk[eleccion[2]]
                j2 = dk[eleccion[3]]

                #print(i1,j1,i2,j2)

                #if tensor[eleccion[0],eleccion[1]] == 0 or tensor[eleccion[2],eleccion[3]] == 0:
                if current_graph_molecule.has_edge(i1,j1) == False or current_graph_molecule.has_edge(i2,j2) == False:
                    print("no edge", flush=True)
                    continue

                # Add the initial SMILES to the set
                current_mol = nx_to_rdkit(current_graph_molecule, False)
                current_smiles = Chem.MolToSmiles(current_mol, allHsExplicit=False)
                all_smiles_molecule.add(current_smiles)
                #print(current_smiles)
                current_graph_molecule.remove_edge(i1,j1)
                current_graph_molecule.remove_edge(i2,j2)
                current_graph_molecule.add_edge(i1,i2)
                current_graph_molecule.add_edge(j1,j2)
                #print(current_graph_molecule.has_edge(i1,i2))
                #print(current_graph_molecule.has_edge(j1,j2))

                # Generate the modified SMILES
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
    #end = time.time()  
    #print(f"P: {idp},Sample step took: {end - start} using {attempts} attempts and {componentes_act} components", flush=True   )

    tf, _, _ = embed_edges_manuel(final_graph, list(final_graph.nodes()))
    return tf, modified_smiles
