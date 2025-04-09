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

    # Move the model to the device
    model = model.to(device)
    time_model = time_model.to(device) 

    # initialize the dataframe
    df_generated_temporal = pd.DataFrame()

    # get the valid graph and edge
    valid_graph_b, valid_edge_b, smiles, _ = conjunto

    # here we add a noise for each example
    sigma_list = [0.5] * valid_edge_b.size(0)

    # initialize the valid noise edge list
    valid_noise_edge_b_list = []
    num_swaps = []
    count=0 

    # iterate over the sigma list
    for cuenta_smiles, sigma_i in enumerate(sigma_list):

        # determine the number of swaps
        num_cambios = math.ceil(sigma_i * torch.sum(valid_edge_b[count]).item() / 2)

        # perform double edge swaps
        numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc, cre_acc = connected_double_edge_swap(deepcopy(valid_graph_b[count]), num_cambios, seed = random.Random())
        
        # append the number of swaps
        num_swaps.append(numswaps)

        # embed the edges
        ruido, _, _ = embed_edges_manuel(g_ruido, list(g_ruido.nodes()))
        valid_noise_edge_b_list.append(ruido.clone())
        count=count+1

    # initialize the variables
    molecule_counter = 0 # counter for the molecules
    num_swaps_max = max(num_swaps) # maximum number of swaps for the current batch
    all_smiles_for_all_molecules = [] # this is a list of sets of smiles for each molecule
    best_time_all_molecules = [0.5] * len(num_swaps) # should initialize with all 0.5s
    best_tensor_all_molecules = [None] * len(num_swaps) # should initialize with all None

    # iterate over the molecules
    while molecule_counter < num_swaps_max:

        print(f"Processing molecules step {molecule_counter} of {num_swaps_max}")

        # if the molecule counter is 0, we initialize the time predictions and the tensors
        if molecule_counter == 0:
            time_predictions = [0.5] * len(num_swaps)
            tensors_allmolecules = valid_noise_edge_b_list 
            # put a open set for each molecule
            for i in range(len(num_swaps)):
                all_smiles_for_all_molecules.append(set())
        
        # calculate the data for each molecule
        futures = [executor_gpu.submit(calculate_data_molecule, graph, tensor, num_swaps, contador_molecula) for graph, tensor, num_swaps in zip(valid_graph_b,tensors_allmolecules, num_swaps)]
        
        # Get the results
        results = [future.result() for future in futures]

        # Initialize the dataset
        dataset = []

        # Iterate over the results and get the data
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
        
        # update the previous best time and tensor for each molecule
        for i in range(len(num_swaps)):
            if molecule_counter == 0 or (molecule_counter < num_swaps[i] and time_predictions[i] < best_time_all_molecules[i]):
                best_time_all_molecules[i] = time_predictions[i]
                best_tensor_all_molecules[i] = tensors_allmolecules[i]

        molecule_counter += 1 # next step
        
        # sample the next step graph for each molecule
        results = []
        futures = [executor_gpu.submit(sample_step_graph, graph, tensor.clone(), probs_quadrupletas.detach(), all_smiles_molecule, idp, num_swaps[idp], molecule_counter-1) for graph, tensor, probs_quadrupletas, all_smiles_molecule, idp in zip(valid_graph_b, tensors_allmolecules, probs_quadrupletas_mod_list, all_smiles_for_all_molecules, range(len(num_swaps)))]

        # Get the results
        results = [future.result() for future in futures]

        # Iterate over the results
        for index, result in enumerate(results):
            tensor, smiles = result
            if smiles is not None: # if smiles is not None, add the smiles to the set, if is None, the molecule has been generated
                all_smiles_for_all_molecules[index].add(smiles)
                tensors_allmolecules[index] = tensor
            else:
                continue
                
       
    # After the loop, find the molecule with the smallest time prediction, excluding the original molecule
    for i in range(len(num_swaps)):
        best_tensor = best_tensor_all_molecules[i]

        # Use the best molecule for further processing
        g_gen = components_to_graph(valid_graph_b[i].nodes(data=True), best_tensor)

        # Convert the molecule to RDKit format without hydrogens
        mol_des = nx_to_rdkit(g_gen, False)

        # Add the SMILES and descriptors to the DataFrame
        smiles_str = Chem.MolToSmiles(mol_des)

        # state that the molecule with the smile has been generated
        print(smiles_str)

        # append the smiles to the dataframe
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_des)
        num_atoms = g_gen.number_of_nodes()
        df_generated_temporal = df_generated_temporal._append({'smiles': smiles_str, 'molecular_formula': formula, 'num_atoms': num_atoms}, ignore_index=True)
    
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
    

    # read all smiles that pass the filters
    df = pd.read_csv('Data/molecular_formulas.csv')
    df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

    # get the smiles from the dataframe
    selected_smiles = df['smiles'].tolist()

    # set the date for the molecule generation
    current_date = datetime.now().strftime("%Y%m%d")
    #str_date = current_date
    str_date = 'Prueba_CoCoGraph_2'

    # Create directories if they don't exist
    molsgen_dir = os.path.join("mols_gen", str_date)
    molsgen_dir_temporal = os.path.join("mols_gen", str_date, "temporal_dfs")

    os.makedirs(molsgen_dir, exist_ok=True)
    os.makedirs(molsgen_dir_temporal, exist_ok=True)

    # load the models
    model = GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2()
    time_model = TimePredictionModel_graph()

    # try to load the models
    try:
        checkpoint = torch.load(f'models/241030_allmolecules_ffnet/model_epoch_2_slice_22.pth')
        checkpoint_time = torch.load(f'models/241017_all_timepred/model_epoch_2_slice_22.pth')
    except:
        print("Could not find the models")
        exit()
    

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']

    if checkpoint_time is not None:
        time_model.load_state_dict(checkpoint_time['model_state_dict'])
        epoch = checkpoint['epoch']

    # set the models to evaluation mode
    model.eval()
    time_model.eval()
    

    # Process smiles in batches of 1000
    batch_size = 1000
    initial_range = 0
    quantity = 50 # molecules per batch
    batches_to_store = 50  # 50K molecules per save (50 batches * 1000 molecules)
    total_batches_processed = 0
    df_generated_total = pd.DataFrame() # dataframe to store all generated molecules
    while True:  # Run indefinitely
        # Randomly sample batch_size SMILES with replacement to generate every time a different set of molecules
        batch_smiles = np.random.choice(selected_smiles, size=batch_size, replace=True)
        
        # Prepare the geometric pytorch data loaders for this batch
        train_dl, validation_dl, test_dl, bonds_perc = build_dataset_alejandro(
            batch_smiles,
            ftr=1.0, fva=0.0,
            bs=quantity,
            min_atom=5,
        )

        # initialize the dataframes
        df_generated_batch_slice = pd.DataFrame()
        df_generated_batch = pd.DataFrame()
        with ProcessPoolExecutor(max_workers=8) as executor_gpu:
            
            # Map process_batch function over all batches   
            for num, conjunto in enumerate(train_dl):
                df_generated_batch_slice = process_batch(conjunto, model, num, total_batches_processed, quantity, time_model)
                df_generated_batch_slice.to_csv(f"mols_gen/{str_date}/batch_{total_batches_processed}_{num}_generated_molecules.csv", index=False) # save the generated molecules for each batch
                df_generated_batch = df_generated_batch._append(df_generated_batch_slice, ignore_index=True)
        
        # Save intermediate results
        df_generated_batch.to_csv(f"mols_gen/{str_date}/generated_molecules_batch_{i}.csv", index=False)

        df_generated_total = df_generated_total._append(df_generated_batch, ignore_index=True) # append the generated molecules to the total dataframe
        
        # Clear memory
        del train_dl, validation_dl, test_dl
        gc.collect()

        total_batches_processed += 1

        # Save total results every batches_to_store batches 
        if total_batches_processed % batches_to_store == 0:
            current_molecules = total_batches_processed * batch_size
            df_generated_total.to_csv(f"mols_gen/{str_date}/all_generated_molecules.csv", index=False)
            print(f"Saved {current_molecules} molecules")
