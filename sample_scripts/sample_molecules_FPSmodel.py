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
from lib_functions.adjacency_utils import nx_to_rdkit
from lib_functions.adjacency_utils import connected_double_edge_swap

from rdkit import Chem
from copy import deepcopy
import gc
import argparse
import random 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def process_batch(conjunto, model, num, b_molecule, cantidad, time_model):

    # Move the model to the device
    model = model.to(device)
    time_model = time_model.to(device) 

    # Initialize the dataframe
    df_generated_temporal = pd.DataFrame()

    # Get the valid graph and edge
    valid_graph_b, valid_edge_b, smiles, _ = conjunto

    # here we add a noise for each example
    sigma_list = [0.5] * valid_edge_b.size(0)

    # Initialize the valid noise edge list
    valid_noise_edge_b_list = []
    num_swaps = []
    count=0 

    # Iterate over the sigma list
    for cuenta_smiles, sigma_i in enumerate(sigma_list):

        # Determine the number of swaps
        num_cambios = math.ceil(sigma_i * torch.sum(valid_edge_b[count]).item() / 2)

        # Perform double edge swaps
        numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc, cre_acc = connected_double_edge_swap(deepcopy(valid_graph_b[count]), num_cambios, seed = random.Random())
        
        # Append the number of swaps
        num_swaps.append(numswaps)

        # Embed the edges
        ruido, _, _ = embed_edges_manuel(g_ruido, list(g_ruido.nodes()))
        valid_noise_edge_b_list.append(ruido.clone()) # store the adjacency matrix of the noisy graph
        count=count+1

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
            tensors_allmolecules = valid_noise_edge_b_list 
            # put a open set for each molecule
            for i in range(len(num_swaps)):
                all_smiles_for_all_molecules.append(set())
        
        # Calculate the data for each molecule at current step
        futures = [executor_gpu.submit(calculate_data_molecule_fps, graph, tensor, num_swaps, molecule_counter) for graph, tensor, num_swaps in zip(valid_graph_b, tensors_allmolecules, num_swaps)]

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
        # get the best molecule for each molecule
        best_tensor = best_tensor_all_molecules[i]

        # Use the best molecule for further processing
        g_gen = components_to_graph(valid_graph_b[i].nodes(data=True), best_tensor)

        # Convert the molecule to RDKit format without hydrogens
        mol_des = nx_to_rdkit(g_gen, False)

        # Add the SMILES and descriptors to the DataFrame
        smiles_str = Chem.MolToSmiles(mol_des)

        # append the smiles to the dataframe
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_des)
        num_atoms = g_gen.number_of_nodes()
        df_generated_temporal = df_generated_temporal._append({'smiles': smiles_str, 'molecular_formula': formula, 'num_atoms': num_atoms}, ignore_index=True)

    return df_generated_temporal
    


if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Molecular generation script using CoCoGraph.')
    parser.add_argument('--input_smiles_csv', type=str, default='Data/molecular_formulas.csv', help='Path to the input CSV file containing SMILES.')
    parser.add_argument('--output_dir_suffix', type=str, default='Prueba_CoCoGraph_2', help='Suffix for the output directory name.')
    parser.add_argument('--model_checkpoint_path', type=str, default='models/241213_allmolecules_ffnet_fps_finetune/model_epoch_1_slice_22.pth', help='Path to the main model checkpoint.')
    parser.add_argument('--time_model_checkpoint_path', type=str, default='models/241216_all_timepred_fps_finetune/model_epoch_2_slice_22.pth', help='Path to the time prediction model checkpoint.')
    parser.add_argument('--batch_size_sample', type=int, default=1000, help='Number of SMILES to sample from the input file in each main loop iteration.')
    parser.add_argument('--batch_size_process', type=int, default=50, help='Batch size for processing molecules within process_batch.')
    parser.add_argument('--save_every_n_batches', type=int, default=50, help='Save cumulative results every N sampling batches.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for parallel execution.')
    
    # Parse the arguments
    args = parser.parse_args()

    # Set the start method for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # read all smiles that pass the filters
    df = pd.read_csv(args.input_smiles_csv)
    df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

    # get the smiles from the dataframe
    selected_smiles = df['smiles'].tolist()

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
    while True:  # Run indefinitely
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
                # Pass args.batch_size_process explicitly to process_batch if needed,
                # or ensure process_batch uses the 'cantidad' parameter correctly
                # Currently 'cantidad' is passed but not used inside process_batch
                future = executor_gpu.submit(process_batch, conjunto, model, num, total_batches_processed, quantity, time_model)
                futures.append(future)

            for i, future in enumerate(futures):
                df_generated_batch_slice = future.result()
                batch_num_within_epoch = i # Use loop index for clarity
                df_generated_batch_slice.to_csv(f"{molsgen_dir_temporal}/batch_{total_batches_processed}_{batch_num_within_epoch}_generated_molecules.csv", index=False) # save the generated molecules for each batch slice
                df_generated_batch = df_generated_batch._append(df_generated_batch_slice, ignore_index=True) # append the generated molecules to the total dataframe

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
