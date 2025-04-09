from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import TimePredictionModel_graph, TimePredictionModel_graph_fps_finetune
from lib_functions.data_preparation_utils import embed_graph_nodes_norm_timepred
from lib_functions.data_preparation_utils import compute_features_fps, compute_features_cero_fps, compute_features_timepred, save_plot_data
from lib_functions.adjacency_utils import connected_double_edge_swap

from lib_functions.data_loader import build_dataset_alejandro

import random 
import os
from copy import deepcopy
import concurrent.futures
import gc 
import argparse

def main(train_dl,  model, checkpoint, executor, slice, epoch, optimizers, schedulers):

    # Get the optimizers and schedulers
    optimizer_pretrained, optimizer_new = optimizers
    scheduler_pretrained, scheduler_new = schedulers

    # Print optimizer's state_dict #hacer esto en un modelo de 0 y en uno cargado que haya modificado el scheduler
    print("Optimizer's state_dict:")
    for var_name in optimizer_pretrained.state_dict():
        print(var_name, "\t", optimizer_pretrained.state_dict()[var_name])

    # Print scheduler's state_dict
    print("Scheduler's state_dict:")
    print(scheduler_pretrained.state_dict())

    # Step the scheduler
    scheduler_pretrained.step(scheduler_pretrained.state_dict()['_step_count']-1)
    scheduler_new.step(scheduler_new.state_dict()['_step_count']-1)

    # Print the current learning rate
    for param_group in optimizer_pretrained.param_groups:
        print("Current Learning Rate:", param_group['lr'])

    # Initialize lists for storing mean losses
   
    trlosses = []
    lm1_l,  lm1_l_b  = [],[]
    batch_maximos = 0  
    epoch_max = epoch+1

    # Define the criterion
    criterion = nn.MSELoss() 

    # Train the model
    while epoch < epoch_max:
        model.train()
        batch = 0

        # Iterate over the training data
        for train_graph_b, train_edge_b, smiles, atoms in train_dl:
            print("batch", batch)

            # Start the timer
            startt = datetime.now()
            
            # Generate a noise level for each example
            sigma_list = [0.5] * train_edge_b.size(0)

            count=0 
            
            # Iterate over the sigma list
            for sigma_i in sigma_list:
                dataset = [] # Initialize the dataset
                nls = [] # Initialize the noise level list
                nls_real = [] # Initialize the real noise level list

                # Determine the number of swaps
                num_swaps = math.ceil(sigma_i * torch.sum(train_edge_b[count]).item() / 2)

                # Perform double edge swaps
                numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc , cre_acc = connected_double_edge_swap(deepcopy(train_graph_b[count]), num_swaps, seed = random.Random())

                # Check if the edge swaps are complete
                if end == True: 
                    count += 1
                    continue

                # Submit tasks
                try: # in case of error, shut down the executor and restart it

                    futures = [executor.submit(compute_features_fps, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]
                    
                    
                    # Collect results
                    results = []
                    for future in futures:
                        results.append(future.result(timeout=60))

                    # get the first graph features
                    ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd, fingerprint = compute_features_cero_fps(train_graph_b[count])
                    distances = torch.Tensor(distances)
                    dosd = torch.Tensor(dosd)
                    fingerprint = torch.Tensor(fingerprint)
                    # append to dataset
                    dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=0.0, distances=distances, final_entrada = ruido, dosd_distances = dosd, morgan_fp = fingerprint))    

                    # Iterate over the results
                    for result, contador_molecula in zip(results, range(len(results))):
                        ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd, fingerprint = result
                        nl = torch.tensor(sigma_i/numswaps * (contador_molecula + 1))
                        nls.append(nl)
                        distances = torch.Tensor(distances)
                        dosd = torch.Tensor(dosd)
                        fingerprint = torch.Tensor(fingerprint)
                        # append to dataset
                        dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=nl, distances=distances, final_entrada = ruido, dosd_distances = dosd, morgan_fp = fingerprint))
                        nls = torch.Tensor(nls).to(device)
                    
                    # Zero the gradients
                    optimizer_pretrained.zero_grad()
                    optimizer_new.zero_grad()

                    # Forward pass
                    predictions = []
                    for graph_data in dataset:
                        graph_data = graph_data.to(device)

                        # Forward pass
                        prediction = model(graph_data)
                        predictions.append(prediction)

                    # Stack the predictions and squeeze the extra dimensions
                    predictions = torch.stack(predictions).squeeze(1).squeeze(1)
 
                    # Compute the loss
                    loss = criterion(predictions, nls)
                    
                    # Backward pass
                    loss.backward()

                    # Update the parameters
                    optimizer_pretrained.step()
                    optimizer_new.step()

                    # Append the loss to the list
                    loss_molec = loss.detach().cpu().item()
                    lm1_l_b.append(loss_molec)
                except Exception as e: # if there is an error, shut down the executor and restart it
                    executor.shutdown(wait=True)  # Shut down the broken executor
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=24)
                    print(f"\033[31m {e} \033[0m")
                    continue
                
            # One batch has been processed
            batch +=1
            batch_maximos +=1
            
            # Compute the loss for the batch
            loss_batch = np.mean(lm1_l_b)
            lm1_l.append(loss_batch)
            
            # Print the time for the batch
            finalt = datetime.now()
            print("dif total:",finalt-startt )
            print("loss",loss_batch)
            print("--------------------------------")
            
            # Garbage collection
            if (batch_maximos % 5) == 0:
                gc.collect()

            # Save the model
            if (batch_maximos % 1000) == 0:
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_pretrained_state_dict': optimizer_pretrained.state_dict(),
                    'optimizer_new_state_dict': optimizer_new.state_dict(),
                    'scheduler_pretrained_state_dict': scheduler_pretrained.state_dict(),
                    'scheduler_new_state_dict': scheduler_new.state_dict(),
                    'loss': loss}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.pth')
                
                # Step the scheduler
                scheduler_pretrained.step()

                # Compute the loss for the epoch
                loss_epoch = np.mean(lm1_l)
                trlosses.append(loss_epoch)

                # To see the current learning rate:
                for param_group in optimizer_pretrained.param_groups:
                    print("Current Learning Rate:", param_group['lr'])

                # Save the losses
                plot_data = {
                    'Train loss MiniB': lm1_l
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/batch_loss_timepred_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
    
                plot_data = {
                    'Train loss': trlosses,
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_timepred_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
                
                lm1_l, lm1_l_b  = [], []
                
        # Save the model
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_pretrained_state_dict': optimizer_pretrained.state_dict(),
                    'optimizer_new_state_dict': optimizer_new.state_dict(),
                    'scheduler_pretrained_state_dict': scheduler_pretrained.state_dict(),
                    'scheduler_new_state_dict': scheduler_new.state_dict(),
                    'loss': loss}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}.pth')
        # scheduler_y.step(epoch)
        # scheduler_z.step(epoch)

        # Compute the loss for the epoch
        loss_epoch = np.mean(lm1_l)
        print("epoch_loss",loss_epoch)
        trlosses.append(loss_epoch)

        # Save the losses
        plot_data = {
            'Train loss MiniB': lm1_l
        }
        save_plot_data(plot_data, f"files/{date_str}/resultados/batch_loss_timepred_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")

        plot_data = {
            'Train loss': trlosses,
        }
        save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_timepred_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
                
        
        lm1_l, lm1_l_b = [], []
        trlosses, trlosses_b_l = [], []
        epoch +=1
    return batch_maximos

def initialize_finetune_model(pretrained_model_path, device):
    # Initialize the Morgan-enhanced model
    finetune_model = TimePredictionModel_graph_fps_finetune().to(device)
    
    # Load the pre-trained model
    pretrained_model = TimePredictionModel_graph()
    pretrained_checkpoint = torch.load(pretrained_model_path, map_location=device)
    pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

    # Get state dicts
    finetune_state_dict = finetune_model.state_dict()
    pretrained_state_dict = pretrained_model.state_dict()
    
    # Filter out keys that are common between the two models
    common_keys = set(finetune_state_dict.keys()).intersection(set(pretrained_state_dict.keys()))
    pretrained_common_dict = {k: v for k, v in pretrained_state_dict.items() if k in common_keys}
    print(pretrained_common_dict.keys())

    # Update the finetune model's state dict with the pre-trained weights
    finetune_state_dict.update(pretrained_common_dict)
    finetune_model.load_state_dict(finetune_state_dict)
    
    # Confirm that the pre-trained model has been loaded successfully by checking a sample weight
    sample_key = next(iter(pretrained_common_dict))
    print(f"Pre-trained model loaded successfully. Sample weight for '{sample_key}': {pretrained_common_dict[sample_key][:5]}")
    print("Proceeding to finetune the model.")

    return finetune_model

def initialize_optimizers(model):
    # Separate parameters: pre-trained and new
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        if 'morgan_fp_mlp' in name or 'concatenate_morgan_fp' in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    print(len(pretrained_params), len(new_params))
    
    # Define optimizers with different learning rates if desired
    optimizer_pretrained = torch.optim.Adam(pretrained_params, lr=1e-5)  # Lower LR for pre-trained
    optimizer_new = torch.optim.Adam(new_params, lr=1e-4)  # Higher LR for new layers
    
    # Learning rate schedulers
    scheduler_pretrained = torch.optim.lr_scheduler.ExponentialLR(optimizer_pretrained, gamma=0.995)
    scheduler_new = torch.optim.lr_scheduler.ExponentialLR(optimizer_new, gamma=0.995)
    
    return optimizer_pretrained, optimizer_new, scheduler_pretrained, scheduler_new

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for molecule processing')
    parser.add_argument('--num_molecules', type=int, default=30000, help='Number of molecules to process')
    parser.add_argument('--slice', type=int, default=0, help='Slice of the big process')
    parser.add_argument('--epoch', type=int, default=0, help='Epoch of training')

    # Parse the arguments
    args = parser.parse_args()

    # Load the dataframe
    with open('Data/TotalSmilesTogether.pickle', 'rb') as inf:
        df = load(inf)
        df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

    # Define the start and end index    
    start_index = args.start_index
    end_index = start_index + args.num_molecules
    selected_smiles = df.smiles.unique()[start_index:end_index]

    # Prepare the geometric pytorch data loaders
    train_dl, validation_dl, test_dl, bonds_perc= build_dataset_alejandro(
        selected_smiles,
        ftr=FTR, fva=FVA,
        bs=12,
        min_atom=5, 
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Get the current date
    current_date = datetime.now() 

    # Convert the date to the desired format
    #date_str = current_date.strftime('%y%m%d')
    date_str = "Prueba_Finetune_timepred_fps"
    
    # Create directories if they don't exist
    files_dir = os.path.join("files", date_str)
    models_dir = os.path.join("models", date_str)
    resultados_dir = os.path.join(f"files/{date_str}/resultados", date_str)

    # Create directories if they don't exist
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
   
    # Load the checkpoint if available
    if (args.epoch == 0) & (args.slice == 0):
        checkpoint = None
    else:
        if args.slice ==0:
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch-1}_slice_{22}.pth')
        else: 
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch}_slice_{args.slice-1}.pth')
        
    # Path to your pre-trained model
    pretrained_model_path = 'models/241017_all_timepred/model_epoch_0_slice_22.pth'
    # Initialize finetune model with pre-trained weights
    finetune_model = initialize_finetune_model(pretrained_model_path, device)

    # Move the model to the device
    model = finetune_model.to(device)

    # Initialize the executor
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:

        # Initialize optimizers
        optimizer_pretrained, optimizer_new, scheduler_pretrained, scheduler_new = initialize_optimizers(finetune_model)
        
        # Load the checkpoint if available
        if checkpoint is not None:
            finetune_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_pretrained.load_state_dict(checkpoint['optimizer_pretrained_state_dict'])
            optimizer_new.load_state_dict(checkpoint['optimizer_new_state_dict'])
            scheduler_pretrained.load_state_dict(checkpoint['scheduler_pretrained_state_dict'])
            scheduler_new.load_state_dict(checkpoint['scheduler_new_state_dict'])

        # Train the model
        main(train_dl, model, checkpoint, executor, args.slice, args.epoch, 
        optimizers=(optimizer_pretrained, optimizer_new), schedulers=(scheduler_pretrained, scheduler_new))
