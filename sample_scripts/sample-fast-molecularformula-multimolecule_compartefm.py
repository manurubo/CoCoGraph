import pickle
import gc
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2, TimePredictionModel_graph
from lib_functions.sample_utils import calculate_data_molecule, sample_step_graph
from lib_functions.data_preparation_utils import embed_edges_manuel
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
