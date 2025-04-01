import pickle
import gc
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2, TimePredictionModel_graph
from lib_functions.data_preparation_utils import embed_edges_with_cycle_sizes_norm, smiles_to_graph
from lib_functions.data_preparation_utils import embed_graph_nodes_norm
from lib_functions.data_preparation_utils import count_cycles_by_size
from lib_functions.data_preparation_utils import embed_edges_manuel
from lib_functions.data_preparation_utils import calculate_2d_distances_ordered
from lib_functions.data_loader import build_dataset_alejandro
from lib_functions.adjacency_utils import components_to_graph
from lib_functions.adjacency_utils import nx_to_rdkit
from rdkit import Chem
from copy import deepcopy

import random 

import os

from lib_functions.adjacency_utils import connected_double_edge_swap

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import multiprocessing

from concurrent.futures import ProcessPoolExecutor

import time

filter_matrix = torch.ones((MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM))
ind = torch.cartesian_prod(torch.arange(MAX_ATOM), torch.arange(MAX_ATOM), torch.arange(MAX_ATOM), torch.arange(MAX_ATOM))
unique_mask = (ind[:, 0] != ind[:, 1]) & \
                            (ind[:, 0] != ind[:, 2]) & \
                            (ind[:, 0] != ind[:, 3]) & \
                            (ind[:, 1] != ind[:, 2]) & \
                            (ind[:, 1] != ind[:, 3]) & \
                            (ind[:, 2] != ind[:, 3])
filter_matrix = unique_mask.reshape((MAX_ATOM, MAX_ATOM, MAX_ATOM, MAX_ATOM)).float().to(device)



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



def sample_step(current_graph_molecule, tensor, probs_quadrupletas_mod, all_smiles_molecule, idp, num_swaps, contador_molecula):
    if contador_molecula >= num_swaps:
        return tensor, None
    start = time.time()
    tensor_copy = tensor.clone()
    end = time.time()
    #print(f"P: {idp},Copy tensor took: {end - start}", flush=True)

    start = time.time()
    mask_des = (tensor.to(device) > 0.5).int() 
    mask_haz = (tensor.to(device) < 2.5).int()
    mask_quads = mask_des.unsqueeze(2).unsqueeze(3) * mask_des.unsqueeze(0).unsqueeze(1) * mask_haz.unsqueeze(1).unsqueeze(3) * mask_haz.unsqueeze(0).unsqueeze(2)

    mask_quads = mask_quads
    end = time.time()
    #print(f"P: {idp},Mask quads took: {end - start}", flush=True)

    start = time.time()
    probs_quadrupletas_mod = probs_quadrupletas_mod.to(device) * mask_quads
    end = time.time()
    #print(f"P: {idp},Probs quads mod took: {end - start}", flush=True)

    start = time.time()
    probs_quadrupletas_mod = probs_quadrupletas_mod * filter_matrix
    end = time.time()
    #print(f"P: {idp},Filter matrix took: {end - start}", flush=True)

    start = time.time()
    flat_prob_tensor = probs_quadrupletas_mod.flatten().double()
    end = time.time()
    #print(f"P: {idp},Flatten took: {end - start}", flush=True)

    start = time.time()
    lim_prob = 0.95
    flat_prob_tensor[flat_prob_tensor < lim_prob] = 0
    end = time.time()
    #print(f"P: {idp},Lim prob took: {end - start}", flush=True)

    start = time.time()
    cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)
    cumulative_distribution2 = cumulative_distribution.clone()
    cumulative_distribution2 /= cumulative_distribution[-1]
    end = time.time()
    #print(f"P: {idp},Cumulative distribution took: {end - start}", flush=True)


    roto = False
    attempts = 0  # Counter for attempts in a single step
    max_attempts = 50 # si lo dejo asi da resultados igual de buenos?
    rompia = 0
    igual_molecula = 0
    start = time.time()
    while not roto and attempts < max_attempts:

            # 3. Utilizar torch.multinomial para muestrear una cuádrupleta basada en la distribución de probabilidad
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
                #attempts += 1

            else:
                eleccion = sampled_position

                i1 = eleccion[0]
                j1 = eleccion[1]
                i2 = eleccion[2]
                j2 = eleccion[3]
    
                if tensor[i1,j1] == 0 or tensor[i2,j2] ==0:
                    continue

                # Add the initial SMILES to the set
                current_mol = nx_to_rdkit(current_graph_molecule, False)
                current_smiles = Chem.MolToSmiles(current_mol, allHsExplicit=False)
                all_smiles_molecule.add(current_smiles)

                print(type(current_graph_molecule))

                tensor[i1, j1] -= 1
                tensor[j1, i1] -= 1
                tensor[i2, j2] -= 1
                tensor[j2, i2] -= 1
    
                tensor[i1, i2] += 1
                tensor[j1, j2] += 1
                tensor[i2, i1] += 1
                tensor[j2, j1] += 1
    
                # Generate the modified SMILES
                modified_graph = components_to_graph(current_graph_molecule.nodes(data=True), tensor)
                modified_mol = nx_to_rdkit(modified_graph, False)
                modified_smiles = Chem.MolToSmiles(modified_mol, allHsExplicit=False)

                processed_graph = components_to_graph(current_graph_molecule.nodes(data=True), tensor)
                componentes_act = nx.number_connected_components(processed_graph)
                
                if (componentes_act <= 1) and (modified_smiles not in all_smiles_molecule):
                    rompia = 0 
                    igual_molecula = 0
                    roto = True
                    
                    final_tensor = tensor.clone()
                    
                   
                else:
                    if (componentes_act > 1):
                        rompia += 1 
                    else:
                        igual_molecula += 1
                    tensor = tensor_copy.clone()
                    attempts += 1
    if attempts >= max_attempts:
        return tensor, None
    end = time.time()
    print(f"P: {idp},Sample step took: {end - start} using {attempts} attempts and {componentes_act} components", flush=True   )


    return final_tensor, modified_smiles

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


def apply_swap_and_count_cycles_g(graph, i1, j1, i2, j2):
    nodos = list(graph.nodes())
    
    graph.remove_edge(nodos[i1], nodos[j1])
    graph.remove_edge(nodos[i2], nodos[j2])
    graph.add_edge(nodos[i1], nodos[i2])
    graph.add_edge(nodos[j1], nodos[j2])
    
    new_graph = nx.Graph(graph)
    
    graph.remove_edge(nodos[i1], nodos[i2])
    graph.remove_edge(nodos[j1], nodos[j2])
    graph.add_edge(nodos[i1], nodos[j1])
    graph.add_edge(nodos[i2], nodos[j2])

    cuenta = count_cycles_by_size(new_graph)

    return cuenta, (i1,j1,i2,j2)

# filling molecule: 'CN(CC(=O)O)C(=N)N ' Creatine hehe
# Create filling values for all variables
processed_graph_filling = smiles_to_graph('CN(CC(=O)O)C(=N)N')

tensor_filling = embed_edges_manuel(processed_graph_filling, list(processed_graph_filling.nodes()))

componentes_ant_filling = nx.number_connected_components(processed_graph_filling)
mol_filling = nx_to_rdkit(processed_graph_filling, True)
mol_filling.UpdatePropertyCache()

gemb_filling, nemb_filling, distances_filling = embed_graph_nodes_norm(processed_graph_filling)

edge_index_filling, edge_attr_filling = embed_edges_with_cycle_sizes_norm(processed_graph_filling)

dosd_positions_filling = calculate_2d_distances_ordered(processed_graph_filling, list(processed_graph_filling.nodes())) 
dosd_positions_filling = torch.Tensor(dosd_positions_filling)

# function to calculate the data for each molecule
# if current_swap is equal or greater than num_swaps, then we can just return None to mark that fill values should be given to data
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



def process_batch(conjunto, model, num, b_molecule, cantidad, time_model):
    model = model.to(device)
    time_model = time_model.to(device) 

    df_generated_temporal = pd.DataFrame()

    valid_graph_b, valid_edge_b, smiles, _ = conjunto
    # aqui saca un ruido para cada ejemplo, la prob de cambio? si
    sigma_list = [0.5] * valid_edge_b.size(0)

    valid_noise_edge_b_list = []
    num_swaps = []
    count=0 

    start_time = time.time()
    # filtro indices no repetidos            
    
    end_time = time.time()
    #print(f"Processing batch {i}, slice {num}: Filtering indices took: {end_time - start_time}")

    
    

    # add noise to the graph
    start_time = time.time()
    for cuenta_smiles, sigma_i in enumerate(sigma_list):

        num_cambios = math.ceil(sigma_i * torch.sum(valid_edge_b[count]).item() / 2)
        numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc, cre_acc = connected_double_edge_swap(deepcopy(valid_graph_b[count]), num_cambios, seed = random.Random())
        
        num_swaps.append(numswaps)
        ruido, _, _ = embed_edges_manuel(g_ruido, list(g_ruido.nodes()))
        valid_noise_edge_b_list.append(ruido.clone())
        count=count+1
    end_time = time.time()
    #print(f"Processing batch {i}, slice {num}: Adding noise to all graphs took: {end_time - start_time}")


    # Doing the cicle for every molecule at the same time
    # important: care about every number of swaps per molecule!
    # easy solution: use the highest number of swaps per molecule
    contador_molecula = 0
    num_swaps_max = max(num_swaps)
    all_smiles_for_all_molecules = [] # this is a list of sets of smiles for each molecule
    best_time_all_molecules = [0.5] * len(num_swaps) # should initialize with all 0.5s
    best_tensor_all_molecules = [None] * len(num_swaps) # should initialize with all None
    molecule_tensors = [[] for _ in range(len(num_swaps))]
    while contador_molecula < num_swaps_max:

        print(f"Processing molecules step {contador_molecula} of {num_swaps_max}")

        if contador_molecula == 0:
            time_predictions = [0.5] * len(num_swaps)
            tensors_allmolecules = valid_noise_edge_b_list # is this well done? or should I make a copy of the list?
            # put a open set for each molecule
            for i in range(len(num_swaps)):
                all_smiles_for_all_molecules.append(set())
        
        # store the whole run of molecule tensors to plot later
        # so we append the tensors to each list
        
        #for i in range(len(num_swaps)):
            
        #    if contador_molecula < num_swaps[i]:
                # Start of Selection
                # Append the tensor to the corresponding molecule's tensor list only
        #        molecule_tensors[i].extend([tensors_allmolecules[i]])
            
        print(f"Calculating data for molecules step {contador_molecula} of {num_swaps_max}")
        # here I calculate the data for each molecule in the current step. Parallelized
        start = time.time()
        # print device of every element
        #print(f"Device of valid_graph_b: {valid_graph_b[0].device}")
        print(f"Device of tensors_allmolecules: {tensors_allmolecules[0].device}")
        #print(f"Device of num_swaps: {num_swaps[0].device}")
        futures = [executor_gpu.submit(calculate_data_molecule, graph, tensor, num_swaps, contador_molecula) for graph, tensor, num_swaps in zip(valid_graph_b,tensors_allmolecules, num_swaps)]
        results = [future.result() for future in futures]
        dataset = []
        for result, prediction_time in zip(results, time_predictions):
            processed_graph, tensor, mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, componentes_ant = result

            d = Data(
                x=nemb,
                edge_index=edge_index,
                y=tensor,
                xA=gemb,
                edge_attr=edge_attr,
                noiselevel=torch.tensor(prediction_time, device=device),
                distances=torch.Tensor(distances),
                dosd_distances=dosd_positions
            ).to(device)
            dataset.append(d)
        end = time.time()
        print(f"Calculating data for molecules step {contador_molecula} of {num_swaps_max} took: {end - start}")

        print(f"Model prediction for molecules step {contador_molecula} of {num_swaps_max}")
        start = time.time()
        # Model prediction for each molecule
        # can it be done like this or should I do it in a loop?
        probs_quadrupletas_mod_list = []
        for d in dataset:
            _,_, probs_quadrupletas_mod = model(d)
            probs_quadrupletas_mod_list.append(probs_quadrupletas_mod)
        end = time.time()
        print(f"Model prediction for molecules step {contador_molecula} of {num_swaps_max} took: {end - start}")

        print(f"Time model prediction for molecules step {contador_molecula} of {num_swaps_max}")
        start = time.time()
        # time model prediction for each molecule
        time_predictions = []
        for d in dataset:
            time_pred = time_model(d)
            time_predictions.append(time_pred.detach().cpu().item())
        end = time.time()
        print(f"Time model prediction for molecules step {contador_molecula} of {num_swaps_max} took: {end - start}")

        print(f"Updating the previous best time and tensor for each molecule step {contador_molecula} of {num_swaps_max}")
        start = time.time()
        # update the previous best time and tensor for each molecule
       
        for i in range(len(num_swaps)):
            if contador_molecula == 0 or (contador_molecula < num_swaps[i] and time_predictions[i] < best_time_all_molecules[i]):
                best_time_all_molecules[i] = time_predictions[i]
                best_tensor_all_molecules[i] = tensors_allmolecules[i]

        end = time.time()
        print(f"Updating the previous best time and tensor for each molecule step {contador_molecula} of {num_swaps_max} took: {end - start}")

        contador_molecula += 1
        
        # We have to do the sampling for each molecule, we can parallelize this
        print(f"Sampling for each molecule step {contador_molecula-1} of {num_swaps_max}")
        start = time.time()
        results = []
        #for graph, tensor, probs_quadrupletas, all_smiles_molecule in zip(valid_graph_b, #tensors_allmolecules, probs_quadrupletas_mod_list, all_smiles_for_all_molecules):
        #    results.append(sample_step(graph, tensor, probs_quadrupletas.detach().cpu(), all_smiles_molecule))

        futures = [executor_gpu.submit(sample_step_graph, graph, tensor.clone(), probs_quadrupletas.detach(), all_smiles_molecule, idp, num_swaps[idp], contador_molecula-1) for graph, tensor, probs_quadrupletas, all_smiles_molecule, idp in zip(valid_graph_b, tensors_allmolecules, probs_quadrupletas_mod_list, all_smiles_for_all_molecules, range(len(num_swaps)))]
        results = [future.result() for future in futures]
        for index, result in enumerate(results):
            tensor, smiles = result
            if smiles is not None: # no more tries on this molecule
                all_smiles_for_all_molecules[index].add(smiles)
                tensors_allmolecules[index] = tensor
                
            else:
                continue
                
        end = time.time()
        print(f"Sampling for each molecule step {contador_molecula-1} of {num_swaps_max} took: {end - start}")
    print(f"Finding the best molecule step {contador_molecula} of {num_swaps_max}")
    start = time.time()
    # After the loop, find the molecule with the smallest time prediction, excluding the original molecule
    for i in range(len(num_swaps)):
        best_tensor = best_tensor_all_molecules[i]

        # Use the best molecule for further processing
        g_gen = components_to_graph(valid_graph_b[i].nodes(data=True), best_tensor)

        # Convert the molecule to RDKit format without hydrogens
        mol_des = nx_to_rdkit(g_gen, False)

        # Dibujar y guardar la molécula sin hidrógenos
        # img = Draw.MolToImage(mol_des)
        # img.save(f'mols_gen/{str_date}/molecule_{b_molecule+num*cantidad+i}_best.png')

        ## plot the tensors for each molecule
        #cuenta_pasos = 0
        #for tensor in molecule_tensors[i]:
        #    img = Draw.MolToImage(nx_to_rdkit(components_to_graph(valid_graph_b[i].nodes(data=True), tensor), False))
        #    img.save(f'mols_gen/{str_date}/molecule_{b_molecule*1000+num*cantidad+i}_tensor_{cuenta_pasos}.png')
        #    cuenta_pasos += 1

        # Añadir el SMILES y descriptores al DataFrame
        smiles_str = Chem.MolToSmiles(mol_des)

        # state that the molecule with the smile has been generated
        print(smiles_str)

        # append the smiles to the dataframe
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_des)
        num_atoms = g_gen.number_of_nodes()
        # aqui añadir las caracteristicas de la molecula
        df_generated_temporal = df_generated_temporal._append({'smiles': smiles_str, 'molecular_formula': formula, 'num_atoms': num_atoms}, ignore_index=True)
    
    end = time.time()
    print(f"Finding the best molecule step {contador_molecula} of {num_swaps_max} took: {end - start}")

    return df_generated_temporal
    

import argparse

if __name__ == "__main__":
    
    # Create the parser
    #parser = argparse.ArgumentParser(description='Process some integers.')
    
    #parser.add_argument('--start_index', type=int, default=0, help='Starting index for molecule processing')
    #parser.add_argument('--num_molecules', type=int, default=30000, help='Number of molecules to process')
    #parser.add_argument('--slice', type=int, default=0, help='Slice of the big process')

    
    # Parse the arguments
    #args = parser.parse_args()
    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    

    # read molecular formulas
    df = pd.read_csv('Data/molecular_formulas.csv')
    df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

    # Get list of unique molecular formulas
    # unique_formulas = df['molecular_formula'].unique()
    
    # # if formula_to_smiles.pkl exists, load it
    # if os.path.exists('formula_to_smiles.pkl'):
    #     with open('formula_to_smiles.pkl', 'rb') as f:
    #         formula_to_smiles = pickle.load(f)
    # else:
    #     # Create a dictionary mapping formulas to SMILES first
    #     formula_to_smiles = {}
    #     for _, row in tqdm(df.iterrows(), desc="Building dictionary"):
    #         formula = row['molecular_formula']
    #         if formula not in formula_to_smiles:
    #             formula_to_smiles[formula] = []
    #         formula_to_smiles[formula].append(row['smiles'])

    #     # store the formula_to_smiles in a pickle file
    #     with open('formula_to_smiles.pkl', 'wb') as f:
    #         pickle.dump(formula_to_smiles, f)

    # Then select random SMILES for each formula
    # selected_smiles = []
    # for formula in tqdm(unique_formulas, desc="Processing unique formulas"):
    #     selected_smiles.append(np.random.choice(formula_to_smiles[formula]))

    # get the smiles from the dataframe
    selected_smiles = df['smiles'].tolist()

    print(len(selected_smiles))

    str_date = 'Prueba_CoCoGraph_2'

    # Create directories if they don't exist
    molsgen_dir = os.path.join("mols_gen", str_date)
    molsgen_dir_temporal = os.path.join("mols_gen", str_date, "temporal_dfs")

    os.makedirs(molsgen_dir, exist_ok=True)
    os.makedirs(molsgen_dir_temporal, exist_ok=True)

    model = GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2()
    time_model = TimePredictionModel_graph()

    df_generated_batch = pd.DataFrame()
    df_generated_total = pd.DataFrame()
    try:
        checkpoint = torch.load(f'models/241030_allmolecules_ffnet/model_epoch_2_slice_22.pth')
        print("modelo_encontrado")
        checkpoint_time = torch.load(f'models/241017_all_timepred/model_epoch_2_slice_22.pth')
        print("modelo_time_encontrado")
    except:
        print("no encontrado")
        checkpoint = None
    

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']

    if checkpoint_time is not None:
        time_model.load_state_dict(checkpoint_time['model_state_dict'])
        epoch = checkpoint['epoch']

    
    model.eval()
    time_model.eval()
    

    # Process smiles in batches of 1000
    batch_size = 1000
    initial_range = 0
    final_range = 50000
    cantidad= 50
    # if stops, we can change the range initial number
    for i in range(initial_range, min(final_range, len(selected_smiles)), batch_size):
        batch_smiles = selected_smiles[i:i+batch_size]
        
        # Prepare the geometric pytorch data loaders for this batch
        train_dl, validation_dl, test_dl, bonds_perc = build_dataset_alejandro(
            batch_smiles,
            ftr=1.0, fva=0.0,
            bs=cantidad,
            min_atom=5,
        )

        
        # Process this batch
        #df_generated_batch = sample(train_dl, model, checkpoint, 0, cantidad, time_model, checkpoint_time, i)
        

        #for num,conjunto in enumerate(dl):
        #    process_batch(conjunto, model, num, i, cantidad, time_model)
        
        df_generated_batch_slice = pd.DataFrame()
        df_generated_batch = pd.DataFrame()
        with ProcessPoolExecutor(max_workers=8) as executor_gpu:
            
            # Map process_batch function over all batches   
            for num, conjunto in enumerate(train_dl):
                df_generated_batch_slice = process_batch(conjunto, model, num, i, cantidad, time_model)
                df_generated_batch_slice.to_csv(f"mols_gen/{str_date}/batch_{i}_{num}_generated_molecules.csv", index=False)
                df_generated_batch = df_generated_batch._append(df_generated_batch_slice, ignore_index=True)
        
        # Save intermediate results
        df_generated_batch.to_csv(f"mols_gen/{str_date}/generated_molecules_batch_{i}.csv", index=False)

        df_generated_total = df_generated_total._append(df_generated_batch, ignore_index=True)
        
        # Clear memory
        del train_dl, validation_dl, test_dl
        gc.collect()
        
    # Save final results
    df_generated_total.to_csv(f"mols_gen/{str_date}/all_generated_molecules.csv", index=False)
