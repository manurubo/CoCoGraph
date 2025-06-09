from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import GINETimePredictor
from lib_functions.data_preparation_utils import compute_features, compute_features_cero, save_plot_data
from lib_functions.adjacency_utils import connected_double_edge_swap
from lib_functions.data_loader import build_dataset_alejandro

import random
import os
from copy import deepcopy
import concurrent.futures
import gc 
import argparse

def main(train_dl, model, checkpoint, executor, slice, epoch):


    # Common parameters for both parts
    common_params = [p for name, p in model.named_parameters() ]
    
    # Create optimizers
    optimizer = torch.optim.Adam(common_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)

    # Load checkpoint if available
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = torch.optim.Adam(common_params, lr=1e-3)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # Print scheduler's state_dict
    print("Scheduler's state_dict:")
    print(scheduler.state_dict())

    # Step the scheduler to the correct step count
    scheduler.step(scheduler.state_dict()['_step_count']-1)

    # To see the current learning rate:
    for param_group in optimizer.param_groups:
        print("Current Learning Rate:", param_group['lr'])


    # Initialize lists for storing mean losses
    trlosses = []
    lm1_l,  lm1_l_b,  = [],[]
    batch_maximos = 0  
    epoch_max = epoch+1

    # Define the criterion
    criterion = nn.MSELoss()

    # Training loop
    while epoch < epoch_max:
        model.train()
        batch = 0

        # Loop through the training data loader
        for train_graph_b, train_edge_b, smiles, atoms in train_dl:
            print("batch", batch)

            # Set the time for the batch
            startt = datetime.now()
            
            # Generate a list of noise levels for each graph
            sigma_list = [0.5] * train_edge_b.size(0)
            
            count=0 
            
            # Loop through the sigma values (over each graph)   
            for sigma_i in sigma_list:
                dataset = [] # Initialize the dataset for each sigma value
                nls = [] # Initialize the list of noise levels

                # Determine the number of swaps to perform
                num_swaps = math.ceil(sigma_i * torch.sum(train_edge_b[count]).item() / 2)

                # Perform the edge swaps
                numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc , cre_acc = connected_double_edge_swap(deepcopy(train_graph_b[count]), num_swaps, seed = random.Random())

                # Check if the edge swaps are complete
                if end == True: 
                    count += 1  
                    continue

                # Submit tasks  
                try: # try in case of error it restarts the executor
                    futures = [executor.submit(compute_features, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]
                    
                    # Collect results
                    results = []
                    for future in futures:
                        results.append(future.result(timeout=60))

                    # get the first graph features
                    ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd = compute_features_cero(train_graph_b[count])
                    distances = torch.Tensor(distances)
                    dosd = torch.Tensor(dosd)
                    # append to dataset
                    dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=0.0, distances=distances, final_entrada = ruido, dosd_distances = dosd))                        
                    # get the rest of the graphs features
                    for result, contador_molecula in zip(results, range(len(results))):
                        ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd = result
                        nl = torch.tensor(sigma_i/numswaps * (contador_molecula + 1))
                        nls.append(nl)
                        distances = torch.Tensor(distances)
                        dosd = torch.Tensor(dosd)
                        # Create a Data element on the dataset for each graph
                        dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=nl, distances=distances, final_entrada = ruido, dosd_distances = dosd))
                    # Convert the list of noise levels to a tensor and move it to the device
                    nls = torch.Tensor(nls).to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass over the whole dataset
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
                    optimizer.step()

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
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.pth')
                
                # Step the scheduler
                scheduler.step()

                # Compute the loss for the epoch
                loss_epoch = np.mean(lm1_l)
                trlosses.append(loss_epoch)

                # To see the current learning rate:
                for param_group in optimizer.param_groups:
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
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}.pth')

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
    with open('Data/training_smiles.pickle', 'rb') as inf:
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
    date_str = "Prueba_Time_funcionaigual"
    
    # Create directories if they don't exist
    files_dir = os.path.join("files", date_str)
    models_dir = os.path.join("models", date_str)
    resultados_dir = os.path.join(f"files/{date_str}/resultados", date_str)

    # Create directories if they don't exist
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
    
    # Create the model
    model = GINETimePredictor() 
    
    # Load the checkpoint if available
    if (args.epoch == 0) & (args.slice == 0):
        checkpoint = None
    else:
        if args.slice ==0:
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch-1}_slice_{22}.pth')
        else: 
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch}_slice_{args.slice-1}.pth')
        
    # Move the model to the device
    model = model.to(device)

    # Create the executor
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        main(train_dl, model, checkpoint, executor, args.slice, args.epoch)
