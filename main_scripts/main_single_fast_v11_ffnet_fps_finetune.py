from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import GINEdgeQuadrupletPredictor, GINEdgeQuadrupletPredictor_MorganFP
from lib_functions.losses import loss_func_vs_inicio
from lib_functions.data_preparation_utils import compute_features_fps, save_plot_data
from lib_functions.adjacency_utils import generate_padding_mask, connected_double_edge_swap
from lib_functions.data_preparation_utils import generate_swap_tensors_optimized
from lib_functions.data_loader import build_dataset_alejandro

import random 
import os
from copy import deepcopy
import concurrent.futures
import gc 
import argparse


def initialize_finetune_model(pretrained_model_path, device):
    # Initialize the Morgan-enhanced model
    finetune_model = GINEdgeQuadrupletPredictor_MorganFP().to(device)
    
    # Load the pre-trained model
    pretrained_model = GINEdgeQuadrupletPredictor()
    pretrained_checkpoint = torch.load(pretrained_model_path, map_location=device)
    pretrained_model.load_state_dict(pretrained_checkpoint['model_state_dict'])

    # Get state dicts
    finetune_state_dict = finetune_model.state_dict()
    pretrained_state_dict = pretrained_model.state_dict()
    
    # Filter out keys that are common between the two models
    common_keys = set(finetune_state_dict.keys()).intersection(set(pretrained_state_dict.keys()))
    pretrained_common_dict = {k: v for k, v in pretrained_state_dict.items() if k in common_keys}

    # Update the finetune model's state dict with the pre-trained weights
    finetune_state_dict.update(pretrained_common_dict)
    finetune_model.load_state_dict(finetune_state_dict)
    
    return finetune_model

def initialize_optimizers(model):
    # Separate parameters: pre-trained and new
    pretrained_params = []
    new_params = []

    
    for name, param in model.named_parameters():
        if 'morgan_fp_mlp' in name or 'pre_reducer_' in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)
    
    # Define optimizers with different learning rates if desired
    optimizer_pretrained = torch.optim.Adam(pretrained_params, lr=1e-5)  # Lower LR for pre-trained
    optimizer_new = torch.optim.Adam(new_params, lr=1e-4)  # Higher LR for new layers
    
    # Learning rate schedulers
    scheduler_pretrained = torch.optim.lr_scheduler.ExponentialLR(optimizer_pretrained, gamma=0.995)
    scheduler_new = torch.optim.lr_scheduler.ExponentialLR(optimizer_new, gamma=0.995)
    
    return optimizer_pretrained, optimizer_new, scheduler_pretrained, scheduler_new

def main(train_dl,  model, checkpoint, executor, slice, epoch, optimizers, schedulers):
    optimizer_pretrained, optimizer_new = optimizers
    scheduler_pretrained, scheduler_new = schedulers
    # Print optimizer's state_dict #hacer esto en un modelo de 0 y en uno cargado que haya modificado el scheduler
    print("Optimizer's state_dict:")
    for var_name in optimizer_pretrained.state_dict():
        print(var_name, "\t", optimizer_pretrained.state_dict()[var_name])

    # Print scheduler's state_dict
    print("Scheduler's state_dict:")
    print(scheduler_pretrained.state_dict())

    # Step the scheduler to the correct step count
    scheduler_pretrained.step(scheduler_pretrained.state_dict()['_step_count']-1)
    scheduler_new.step(scheduler_new.state_dict()['_step_count']-1)

    # To see the current learning rate:
    for param_group in optimizer_pretrained.param_groups:
        print("Current Learning Rate:", param_group['lr'])

    # Initialize lists for storing mean losses
    mean_train_losses,  mean_train_losses1, mean_train_losses2, mean_train_losses3 = [],[],[], []
    trlosses,  trlosses_b, trlosses_b_l = [], [], []
    lm1_l, lm2_l, lm3_l, lm1_l_b, lm1_l_b_l, lm2_l_b, lm2_l_b_l, lm3_l_b, lm3_l_b_l = [],[],[],[],[],[], [], [], []
    batch_maximos = 0 
    epoch_max = epoch+1

    # Training loop
    while epoch < epoch_max:
        model.train()
        batch = 0

        # Loop through the training data loader
        for train_graph_b, train_edge_b, smiles, atoms in train_dl:
            print("batch", batch)
            # set time for the batch
            startt = datetime.now()
            
            # Generate a list of noise levels for each graph
            sigma_list = [0.5] * train_edge_b.size(0)
            # Generate a mask for the training graphs
            train_mask_b = train_edge_b.sum(-1).gt(1e-3).to(dtype=torch.float32)

            count=0 
            
            # Loop through the sigma values (over each graph)
            for sigma_i in sigma_list:
                dataset = [] # Initialize the dataset for each sigma value
                nls = [] # Initialize the list of noise levels
                # determine the number of swaps to perform
                num_swaps = math.ceil(sigma_i * torch.sum(train_edge_b[count]).item() / 2)
                # Perform the edge swaps
                numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc , cre_acc = connected_double_edge_swap(deepcopy(train_graph_b[count]), num_swaps, seed = random.Random())

                # Check if the edge swaps are complete
                if end == True: 
                    count += 1
                    continue # if no swaps were performed, skip to the next graph
                
                # Initialize the matrices for the edge swaps
                matrices_int = []

                # Submit tasks to compute features with fingerprints
                futures = [executor.submit(compute_features_fps, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]

                # Collect results
                try: # try in case of error it restarts the executor
                    results = []
                    for future in futures:
                        results.append(future.result(timeout=20))
                    
                    # Process results
                    for result, molecule_index in zip(results, range(len(results))):
                        ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd, fingerprint = result
                        matrices_int.append(ruido)
                        distances = torch.Tensor(distances)
                        fingerprint = torch.Tensor(fingerprint)
                        nl = torch.tensor(sigma_i/numswaps * (molecule_index + 1))
                        nls.append(nl)
                        dosd = torch.Tensor(dosd)
                        dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=nl, distances=distances, final_entrada = ruido, dosd_distances = dosd, morgan_fp = fingerprint))
    
                    molecule_index = 0
    
                    # Initialize lists for storing scores and probabilities
                    score_des_l, score_haz_l = [],[]
                    quadruple_probs_list = []
                    train_noise_edge_b_list = []
                    quadrupletes_changed= []
                    minib_nls = []
                    dataset_length = len(dataset)
                    batch_size = 2 # number of minibatches

                    # Iterate over the dataset
                    for graph_data, qc, noise_edge, m_nls  in zip(dataset, final_swaps, matrices_int, nls):
                        
                        # Move graph data to the device (e.g., GPU) if available
                        graph_data = graph_data.to(device)
    
                        # Forward pass
                        score_des, score_haz, quads_prob_mod = model(graph_data)
                        quadruple_probs_list.append(quads_prob_mod)
                        
                        # Append scores and probabilities to the lists
                        score_des_l.append(score_des)
                        score_haz_l.append(score_haz)
                        molecule_index +=1
                        quadrupletes_changed.append(qc)
                        train_noise_edge_b_list.append(noise_edge)
                        minib_nls.append(m_nls)

                        # If the molecule index is a multiple of the batch size or the end of the dataset, we process the minibatch to compute the loss
                        if molecule_index%batch_size==0 or molecule_index==dataset_length:
                            
                            # Concatenate the scores and probabilities
                            score_des = torch.cat(score_des_l, dim=0).squeeze(-1)
                            score_haz = torch.cat(score_haz_l, dim=0).squeeze(-1)

                            # Stack the probabilities
                            quadruple_probs = torch.stack(quadruple_probs_list, dim=0)
    
                            # Generate the swap tensors
                            quadruple_tensors = generate_swap_tensors_optimized(quadrupletes_changed, num_nodes = MAX_ATOM, device = device)
                            
                            # Concatenate the noise edge matrices
                            tensor_list = [matrix.unsqueeze(0) for matrix in train_noise_edge_b_list] 

                            # Create the entry mask so that we can compute the loss only on the edges that are present
                            entry_mask = torch.cat(tensor_list, dim = 0)
                            entry_mask = entry_mask.to(device)
                            entry_mask = (entry_mask>0.5).int()

                            entry_mask1 = entry_mask.unsqueeze(-1).unsqueeze(-1)
                            entry_mask2 = entry_mask.unsqueeze(1).unsqueeze(2)
                            entry_mask = entry_mask1 * entry_mask2
                            
                            # Create the mask for the training graphs based on padding
                            masks_b = generate_padding_mask(train_mask_b[count]) 
    
                            # Repeat the mask for the number of noise edges
                            padding_mask = masks_b.repeat(len(train_noise_edge_b_list), 1, 1)
                            
                            padding_mask = padding_mask.to(device)
                
                            padding_mask1 = padding_mask.unsqueeze(-1).unsqueeze(-1)
                            padding_mask2 = padding_mask.unsqueeze(1).unsqueeze(2)
                            padding_mask = padding_mask1*padding_mask2
                            
                            # Create the mask for the training graphs based on padding
                            final_mask = padding_mask * entry_mask
    
                            # Apply the mask to the probabilities and tensors   
                            quadruple_probs = quadruple_probs * final_mask.to(device)
                            quadruple_tensors = quadruple_tensors * final_mask
                            
                            # Flatten the tensors
                            quadruple_probs_flat = quadruple_probs.view(quadruple_probs.size(0), -1)
                            quadruple_tensors_flat = quadruple_tensors.view(quadruple_tensors.size(0), -1)
    
                            # Count the frequency of each class
                            num_ones = (quadruple_tensors_flat == 1).sum()
                            num_zeros = (final_mask.view(final_mask.size(0), -1)==1).sum() - num_ones
                            
                            # Calculate the ratio and scale the weight for 1s
                            ratio = num_zeros.float() / num_ones.float()
                            weight_for_1s = ratio  # Weight for 1s is scaled based on the ratio
                            
                            # Create a weight tensor with scaled weight for 1s and weight 1 for 0s
                            weights = torch.ones_like(quadruple_tensors_flat)
                            weights[quadruple_tensors_flat == 1] = weight_for_1s
    
                            # Define the BCE Loss function
                            criterion = nn.BCELoss(reduction='none')
    
                            # Calculate the loss
                            loss_quadrupletas = criterion(quadruple_probs_flat, quadruple_tensors_flat.to(device)) 
                            weighted_losses_quadrupletas = loss_quadrupletas * weights

                            # Sum the loss over unmasked values
                            final_mask = final_mask.view(final_mask.size(0), -1)
                            loss_quadrupletas_sum = (weighted_losses_quadrupletas * final_mask).sum()

                            # Count the number of unmasked values
                            final_mask_count = final_mask.sum()

                            # Avoid division by zero
                            final_mask_count = torch.clamp(final_mask_count, min=1)
                            loss_quadrupletas = loss_quadrupletas_sum / final_mask_count
    
                            # Create the mask for the training graphs based on padding for the single graph loss
                            padding_mask = masks_b.repeat(len(train_noise_edge_b_list), 1, 1)

                            # Repeat the training edge matrix for the number of noise edges
                            # this sets the objective matrix for the single graph loss
                            final_obj = train_edge_b[count].repeat(len(train_noise_edge_b_list), 1, 1)

                            # Concatenate the noise edge matrices and define the 
                            tensor_list = [matrix.unsqueeze(0) for matrix in train_noise_edge_b_list] 
                            entry_mask = torch.cat(tensor_list, dim = 0)
    
                            # Stack the noise levels
                            final_nl = torch.stack(minib_nls)
                            final_nl = final_nl.unsqueeze(1).unsqueeze(2).repeat(1, MAX_ATOM, MAX_ATOM)

                            # Compute the loss for the single graph
                            lm1, lm2 = loss_func_vs_inicio(score_des, score_haz,final_obj, entry_mask, padding_mask, final_nl)
    
                            # Append the losses to the lists
                            lm1_l.append(lm1.detach().cpu().item())
                            lm2_l.append(lm2.detach().cpu().item())
                            lm3_l.append(loss_quadrupletas.detach().cpu().item())
                            lm1_l_b.append(lm1.detach().cpu().item())
                            lm2_l_b.append(lm2.detach().cpu().item())
                            lm3_l_b.append(loss_quadrupletas.detach().cpu().item())
                            trlosses.append(loss_quadrupletas.detach().cpu().item())
                            trlosses_b.append(loss_quadrupletas.detach().cpu().item())
    
                            # Normalize the losses by the batch size
                            if molecule_index%batch_size == 0:
                                lm1 /= (len(dataset)/batch_size)
                                lm2 /= (len(dataset)/batch_size)
                                loss_quadrupletas /= (len(dataset)/batch_size)
                            else:
                                lm1 /= (len(dataset)/(molecule_index%batch_size))
                                lm2 /= (len(dataset)/(molecule_index%batch_size))
                                loss_quadrupletas /= (len(dataset)/(molecule_index%batch_size))

                            # Backward pass
                            lm1.backward(retain_graph=True)  # Retain the graph for the next backward pass
                            lm2.backward(retain_graph=True)
                            loss_quadrupletas.backward()
    
                            # Clean lists
                            quadrupletes_changed, quadruple_probs_list, train_noise_edge_b_list, minib_nls = [], [], [], []
                            score_des_l, score_haz_l = [],[]
                # If there is an error, shut down the executor and restart it
                except Exception as e:
                    executor.shutdown(wait=True)  # Shut down the broken executor
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=24)
                    print(f"\033[31m {e} \033[0m")
                    continue
                count=count+1
            
            batch +=1
            batch_maximos +=1

            # Step the optimizers
            try:
                optimizer_pretrained.step()
                optimizer_new.step()
                optimizer_pretrained.zero_grad()
                optimizer_new.zero_grad()
            except Exception as e:
                print(f"\033[31m {e} \033[0m")
                continue
            
            # Print the time for the batch
            finalt = datetime.now()
            print("dif total:",finalt-startt )
            # Print the losses
            print("loss q",np.mean(lm3_l_b))
            print("lm1",np.mean(lm1_l_b))
            print("lm2",np.mean(lm2_l_b))
            print("--------------------------------")

            # Append the losses to the lists    
            trlosses_b_l.append(np.mean(trlosses_b))
            trlosses_b = []
            lm1_l_b_l.append(np.mean(lm1_l_b))
            lm1_l_b = []
            lm2_l_b_l.append(np.mean(lm2_l_b))
            lm2_l_b = [] 
            lm3_l_b_l.append(np.mean(lm3_l_b))
            lm3_l_b = [] 

            # Garbage collection
            if (batch_maximos % 5) == 0:
                gc.collect()
            # Save the model every 1000 batches
            if (batch_maximos % 1000) == 0:
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_pretrained_state_dict': optimizer_pretrained.state_dict(),
                    'optimizer_new_state_dict': optimizer_new.state_dict(),
                    'scheduler_pretrained_state_dict': scheduler_pretrained.state_dict(),
                    'scheduler_new_state_dict': scheduler_new.state_dict(),
                    'loss': loss_quadrupletas}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.pth')
                scheduler_pretrained.step()
                scheduler_new.step()
                # To see the current learning rate:
                for param_group in optimizer_pretrained.param_groups:
                    print("Current Learning Rate:", param_group['lr'])

                # Compute the mean losses
                mean_train_loss = np.mean(trlosses)
                mean_train_loss1 = np.mean(lm1_l)
                mean_train_loss2 = np.mean(lm2_l)
                mean_train_loss3 = np.mean(lm3_l)
                mean_train_losses.append(mean_train_loss)
                mean_train_losses1.append(mean_train_loss1)
                mean_train_losses2.append(mean_train_loss2)
                mean_train_losses3.append(mean_train_loss3)
                print("epoch_loss",mean_train_loss)
                # Save the losses
                plot_data = {
                    'Train Deshacer MiniB': lm1_l_b_l,
                    'Train Hacer MiniB': lm2_l_b_l,
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/minibatch_loss_matrixes_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
    
                plot_data = {
                    'Train Quadrups MiniB': lm3_l_b_l,
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/minibatch_loss_quads_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
    
                plot_data = {
                    'Train Deshacer': mean_train_losses1,
                    'Train Hacer': mean_train_losses2,
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_matrixes_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
    
                plot_data = {
                    'Train Quads': mean_train_losses3,
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_quads_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")

                lm1_l, lm2_l, lm3_l, lm1_l_b_l, lm2_l_b_l,  lm3_l_b_l = [],[],[],[],[],[]
                trlosses, trlosses_b_l = [], []
                
        # Save the model at the end of the epoch     
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_pretrained_state_dict': optimizer_pretrained.state_dict(),
                    'optimizer_new_state_dict': optimizer_new.state_dict(),
                    'scheduler_pretrained_state_dict': scheduler_pretrained.state_dict(),
                    'scheduler_new_state_dict': scheduler_new.state_dict(),
                    'loss': loss_quadrupletas}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}.pth')


        # Compute the mean losses
        mean_train_loss = np.mean(trlosses)
        mean_train_loss1 = np.mean(lm1_l)
        mean_train_loss2 = np.mean(lm2_l)
        mean_train_loss3 = np.mean(lm3_l)
        mean_train_losses.append(mean_train_loss)
        mean_train_losses1.append(mean_train_loss1)
        mean_train_losses2.append(mean_train_loss2)
        mean_train_losses3.append(mean_train_loss3)
        print("epoch_loss",mean_train_loss)

        # Save the losses
        plot_data = {
                    'Train Deshacer MiniB': lm1_l_b_l,
                    'Train Hacer MiniB': lm2_l_b_l,
                }
        save_plot_data(plot_data, f"files/{date_str}/resultados/minibatch_loss_matrixes_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")

        plot_data = {
            'Train Quadrups MiniB': lm3_l_b_l,
        }
        save_plot_data(plot_data, f"files/{date_str}/resultados/minibatch_loss_quads_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")

        plot_data = {
            'Train Deshacer': mean_train_losses1,
            'Train Hacer': mean_train_losses2,
        }
        save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_matrixes_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")

        plot_data = {
            'Train Quads': mean_train_losses3,
        }
        save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_quads_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
        
        lm1_l, lm2_l, lm3_l, lm1_l_b_l, lm2_l_b_l,  lm3_l_b_l = [],[],[],[],[],[]
        trlosses, trlosses_b_l = [], []
        epoch +=1
    return batch_maximos


        
if __name__ == "__main__":
    dataloaders_dir = "dataloaders_saved"

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

    # Set the environment variable for the CUDA allocator
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Get the current date
    current_date = datetime.now() 

    # Convert the date to the desired format
    # date_str = current_date.strftime('%y%m%d')  
    date_str = "Prueba_Finetune_ffnet_fps"
    
    # Create directories if they don't exist
    files_dir = os.path.join("files", date_str)
    models_dir = os.path.join("models", date_str)
    resultados_dir = os.path.join(f"files/{date_str}/resultados", date_str)

    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
        
    # Load the checkpoint
    if (args.epoch == 0) & (args.slice == 0):
        checkpoint = None
    else:
        if args.slice ==0:
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch-1}_slice_{22}.pth')
        else: 
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch}_slice_{args.slice-1}.pth')
    if checkpoint is not None:
        print("Checkpoint fields:")
        for key in checkpoint.keys():
            print(f"{key}: {checkpoint[key]}")
        
    # Path to your pre-trained model
    pretrained_model_path = 'models/241030_allmolecules_ffnet/model_epoch_0_slice_22.pth'
    # Initialize finetune model with pre-trained weights
    finetune_model = initialize_finetune_model(pretrained_model_path, device)
    
    # Now proceed with training finetune_model
    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        # Initialize optimizers
        optimizer_pretrained, optimizer_new, scheduler_pretrained, scheduler_new = initialize_optimizers(finetune_model)
        
        if checkpoint is not None:
            finetune_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_pretrained.load_state_dict(checkpoint['optimizer_pretrained_state_dict'])
            optimizer_new.load_state_dict(checkpoint['optimizer_new_state_dict'])
            scheduler_pretrained.load_state_dict(checkpoint['scheduler_pretrained_state_dict'])
            scheduler_new.load_state_dict(checkpoint['scheduler_new_state_dict'])
        
        # Pass multiple optimizers to the training loop
        main(train_dl,  finetune_model, checkpoint, executor, slice=args.slice, epoch=args.epoch,
             optimizers=(optimizer_pretrained, optimizer_new),
             schedulers=(scheduler_pretrained, scheduler_new))