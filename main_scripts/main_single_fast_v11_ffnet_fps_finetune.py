from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2, GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2_morgan_finetune_2
from lib_functions.losses import loss_func_vs_inicio
from lib_functions.data_preparation_utils import compute_features_fps, save_plot_data
from lib_functions.adjacency_utils import generate_mask2, connected_double_edge_swap, nx_to_rdkit
from lib_functions.data_preparation_utils import generate_swap_tensors_optimized

from lib_functions.data_loader import build_dataset_alejandro

import random 
import os
from copy import deepcopy
import concurrent.futures
import itertools
import json
import gc 
from rdkit.Chem import AllChem
from multiprocessing import Pool


def initialize_finetune_model(pretrained_model_path, device):
    # Initialize the Morgan-enhanced model
    finetune_model = GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2_morgan_finetune_2().to(device)
    
    # Load the pre-trained model
    pretrained_model = GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2()
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
    
    #for name, param in model.named_parameters():
    #    if 'morgan_fp_mlp' in name or 'ff_break' in name or 'ff_make' in name or 'reduce_ff' in name:
    #        new_params.append(param)
    #    else:
    #        pretrained_params.append(param)
    
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

def main(train_dl, test_dl, model, checkpoint, executor, slice, epoch, optimizers, schedulers):
    optimizer_pretrained, optimizer_new = optimizers
    scheduler_pretrained, scheduler_new = schedulers
    
    

    # Print optimizer's state_dict #hacer esto en un modelo de 0 y en uno cargado que haya modificado el scheduler
    print("Optimizer's state_dict:")
    for var_name in optimizer_pretrained.state_dict():
        print(var_name, "\t", optimizer_pretrained.state_dict()[var_name])

    # Print scheduler's state_dict
    print("Scheduler's state_dict:")
    print(scheduler_pretrained.state_dict())

    scheduler_pretrained.step(scheduler_pretrained.state_dict()['_step_count']-1)
    scheduler_new.step(scheduler_new.state_dict()['_step_count']-1)
    # To see the current learning rate:
    for param_group in optimizer_pretrained.param_groups:
        print("Current Learning Rate:", param_group['lr'])


    # else: # guardo modelo para poder cargarlo luego con los mismos pesos
    #      torch.save({'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_y_state_dict': optimizer_y.state_dict(),
    #         'optimizer_z_state_dict': optimizer_z.state_dict(),
    #         'loss': 0}, f'models/{date_str}/model_epoch_{epoch}.pth')

    # for name,param in model.named_parameters():
    #     print(name, param)

    print("-------------------------------------------")

    # Initialize lists for storing mean losses
    mean_train_losses,  mean_train_losses1, mean_train_losses2, mean_train_losses3 = [],[],[], []
    mean_valid_losses = []
    trlosses, vlosses, trlosses_b, trlosses_b_l = [], [], [], []
    lm1_l, lm2_l, lm3_l, lm1_l_b, lm1_l_b_l, lm2_l_b, lm2_l_b_l, lm3_l_b, lm3_l_b_l = [],[],[],[],[],[], [], [], []
    batch_maximos = 0  # IMPORTANTE! ajustar por el guardado del modelo 
    epoch_max = epoch+1
    print(epoch, epoch_max)
    while epoch < epoch_max:

        dif_grados = []

        minvloss = 1e100
        trnorm = 0
        model.train()
        batch = 0
        print("epoch",epoch)

        
        # s1 = tracemalloc.take_snapshot()
        for train_graph_b, train_edge_b, smiles, atoms in train_dl:
            print("batch", batch)
            # if batch < 820:
            #     batch+=1
            #     continue

            

            
            print(sum(atoms))
            # print(train_edge_b)
            startt = datetime.now()
            
    #         train_graph_b = train_graph_b.to(device)
            
            # aqui saca un ruido para cada ejemplo, la prob de cambio? si
            sigma_list = [0.5] * train_edge_b.size(0)
            
            train_mask_b = train_edge_b.sum(-1).gt(1e-3).to(dtype=torch.float32)
            

            
       
            count=0 
            
            sumabytes = 0
            for sigma_i in sigma_list:
                dataset = []
                nls = []
                start = datetime.now()
                num_swaps = math.ceil(sigma_i * torch.sum(train_edge_b[count]).item() / 2)
                # print("===========================")
                # sumabytes += asizeof.asizeof(train_graph_b[count])
                numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc , cre_acc = connected_double_edge_swap(deepcopy(train_graph_b[count]), num_swaps, seed = random.Random())

                
                # print(graphs_list)
                # print("===========================")

                # print(rem_acc, cre_acc)
                
                final = datetime.now()
                # print(f'swaps {final-start}')
                if end == True: 
                    count += 1
                    continue
                

                matrices_int = []

                start = datetime.now()
                # Submit tasks
                futures = [executor.submit(compute_features_fps, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]

                # results = []
                # num_e = 0
                # for remove, create in zip(rem_acc, cre_acc):
                #     resultado = compute_features(deepcopy(train_graph_b[count]), num_e, remove)
                #     results.append(resultado)
                #     num_e += 1

                largo_grafos = numswaps

                # start = datetime.now()
                # del(graphs_list)
                # final = datetime.now()
                # print(f"tiempo del {final-start}")
                

                # results = []
                # for graph in graphs_list:
                #     resultado = compute_features(graph)
                #     results.append(res.result())

                # Collect results
                try:
                    results = []
                    for future in futures:
                        results.append(future.result(timeout=20))
                    
                    # Process results
                    # grafos_list= []
                    for result, contador_molecula in zip(results, range(len(results))):
                        ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd, fingerprint = result
                        # grafos_list.append(grafo_i)
                        matrices_int.append(ruido)
                        distances = torch.Tensor(distances)
                        fingerprint = torch.Tensor(fingerprint)
                        nl = torch.tensor(sigma_i/largo_grafos * (contador_molecula + 1))
                        nls.append(nl)
                        dosd = torch.Tensor(dosd)
                        dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=nl, distances=distances, final_entrada = ruido, dosd_distances = dosd, morgan_fp = fingerprint))
    
                    # print("estamos")
                    # print(len(grafos_list),len(graphs_list))
                    # for g2, g3 in zip(grafos_list, graphs_list):
                    #     mol = nx_to_rdkit(g2)
                    #     smiles_str = Chem.MolToSmiles(mol)
                    #     print(smiles_str)
                    #     mol = nx_to_rdkit(g3)
                    #     smiles_str = Chem.MolToSmiles(mol)
                    #     print(smiles_str)
                    # final = datetime.now()
                    # print("dif features y dataset:",final-start )  
                    
                    contador_molecula = 0
    
                    start = datetime.now()
                    score_des_l, score_haz_l = [],[]
                    # Iterate over the graphs in the dataset
                    quadruple_probs_list = []
                    train_noise_edge_b_list = []
                    quadrupletes_changed= []
                    minib_nls = []
                    largo_dataset = len(dataset)
                    corte = 2 # largo_dataset # numero de minibatch
                    
                    for graph_data, qc, noise_edge, m_nls  in zip(dataset, final_swaps, matrices_int, nls):
                        
                        # Move graph data to the device (e.g., GPU) if available
                        graph_data = graph_data.to(device)
    
                        # Forward pass
                        score_des, score_haz, quads_prob_mod = model(graph_data)
                        quadruple_probs_list.append(quads_prob_mod)
                        
    
                        score_des_l.append(score_des)
                        score_haz_l.append(score_haz)
                        contador_molecula +=1
                        quadrupletes_changed.append(qc)
                        train_noise_edge_b_list.append(noise_edge)
                        minib_nls.append(m_nls)
                        if contador_molecula%corte==0 or contador_molecula==largo_dataset:
                            
                            # ahora sacar las cosas, sino siguiente acumulacion
                            score_des = torch.cat(score_des_l, dim=0).squeeze(-1)
                            score_haz = torch.cat(score_haz_l, dim=0).squeeze(-1)
                            # print(score_des.shape)
    
                            quadruple_probs = torch.stack(quadruple_probs_list, dim=0)
    
                            startq = datetime.now()
                            startq2 = datetime.now()
    
                            # quadruple_tensors = generate_swap_tensors(quadrupletes_changed)
                            # quadruple_tensors = torch.stack(quadruple_tensors, dim=0).to(device)
                            quadruple_tensors = generate_swap_tensors_optimized(quadrupletes_changed, num_nodes = MAX_ATOM, device = device)
                            # finalq2 = datetime.now()
                            # print(f'mascaras {finalq2-startq2}')
                            
                            
                            lista_de_tensores = [matrix.unsqueeze(0) for matrix in train_noise_edge_b_list] 
                            final_entrada = torch.cat(lista_de_tensores, dim = 0)
                            final_entrada = final_entrada.to(device)
                            final_entrada = (final_entrada>0.5).int()
                            
                            final_entrada1 = final_entrada.unsqueeze(-1).unsqueeze(-1)
                            final_entrada2 = final_entrada.unsqueeze(1).unsqueeze(2)
                            
                            final_entrada = final_entrada1 * final_entrada2
                            # print(final_entrada.shape)
                            # for qc in quadrupletes_changed:
                            #     print(qc)
                            #     print(final_entrada[0,qc[0],qc[2],qc[1],qc[3]])
                            #     print(final_entrada[1,qc[0],qc[2],qc[1],qc[3]])
                            
                            masks_b = generate_mask2(train_mask_b[count]) # funciona al indexar?
    
                            final_mask = masks_b.repeat(len(train_noise_edge_b_list), 1, 1)
                            
                            final_mask = final_mask.to(device)
                
                            final_mask1 = final_mask.unsqueeze(-1).unsqueeze(-1)
                            final_mask2 = final_mask.unsqueeze(1).unsqueeze(2)
                            final_mask = final_mask1*final_mask2
                            
                            final_mask = final_mask * final_entrada
    
                            # print((quadruple_tensors == 1).sum())
                            quadruple_probs = quadruple_probs * final_mask.to(device)
                            quadruple_tensors = quadruple_tensors * final_mask
                            
                            # print((quadruple_tensors == 1).sum())
    
                            # Assuming quadruple_probs and quadruple_tensors are your tensors
                            # Flatten the tensors
                            quadruple_probs_flat = quadruple_probs.view(quadruple_probs.size(0), -1)
                            quadruple_tensors_flat = quadruple_tensors.view(quadruple_tensors.size(0), -1)
                            
    
                            # Count the frequency of each class
                            num_ones = (quadruple_tensors_flat == 1).sum()
                            num_zeros = (final_mask.view(final_mask.size(0), -1)==1).sum() - num_ones
                            
                            # print(num_ones, num_zeros) # asegurarse que esto esta funcionando bien
    
                            # Calculate the ratio and scale the weight for 1s
                            ratio = num_zeros.float() / num_ones.float()
                            weight_for_1s = ratio  # Weight for 1s is scaled based on the ratio
                            # print(weight_for_1s)
                            
                            # Create a weight tensor with scaled weight for 1s and weight 1 for 0s
                            weights = torch.ones_like(quadruple_tensors_flat)
                            weights[quadruple_tensors_flat == 1] = weight_for_1s
    
                            # Define the BCE Loss function
                            criterion = nn.BCELoss(reduction='none')
    
                            # Calculate the loss
                            # print("pr",quadruple_probs_flat[quadruple_tensors_flat == 1])
                            # print("te",quadruple_tensors_flat[quadruple_tensors_flat == 1])
                            loss_quadrupletas = criterion(quadruple_probs_flat, quadruple_tensors_flat.to(device)) 
                            # print("lq",loss_quadrupletas[quadruple_tensors_flat == 1])
                            # print("we",weights[quadruple_tensors_flat == 1] )
                            weighted_losses_quadrupletas = loss_quadrupletas * weights
                            # print("wlq",weighted_losses_quadrupletas[quadruple_tensors_flat == 1])
                            # Sum the loss over unmasked values
                            final_mask = final_mask.view(final_mask.size(0), -1)
                            # print("mask",final_mask[quadruple_tensors_flat == 1])
                            # print("qc", quadrupletes_changed)
                                
                            loss_quadrupletas_sum = (weighted_losses_quadrupletas * final_mask).sum()
                            # Count the number of unmasked values
                            final_mask_count = final_mask.sum()
                            # print(final_mask_count)
                            # Avoid division by zero
                            final_mask_count = torch.clamp(final_mask_count, min=1)
                            loss_quadrupletas = loss_quadrupletas_sum / final_mask_count
    
                            # finalq = datetime.now()
                            # print(f'dif quads {finalq-startq}')
                                
                                
                            # print(loss_quadrupletas)
                            startm = datetime.now()
    
                            final_mask = masks_b.repeat(len(train_noise_edge_b_list), 1, 1)
                            final_obj = train_edge_b[count].repeat(len(train_noise_edge_b_list), 1, 1)
                            single_graph_matrices = [train_edge_b[count]]
                            for ne in train_noise_edge_b_list[:-1]:
                                single_graph_matrices.append(ne)
                        
                            lista_de_tensores = [matrix.unsqueeze(0) for matrix in single_graph_matrices] # esto ya no funciona exactamente asi, verdad?
                            all_graphs_tensor = torch.cat(lista_de_tensores, dim = 0)
                            # print(all_graphs_tensor.shape)
                            # print(final_mask.shape)
                            # print(final_obj.shape)
    
                            lista_de_tensores = [matrix.unsqueeze(0) for matrix in train_noise_edge_b_list] 
                            final_entrada = torch.cat(lista_de_tensores, dim = 0)
    
                            final_nl = torch.stack(minib_nls)
    
                            final_nl = final_nl.unsqueeze(1).unsqueeze(2).repeat(1, MAX_ATOM, MAX_ATOM)
                            lm1, lm2 = loss_func_vs_inicio(score_des, score_haz,final_obj, final_entrada, final_mask, final_nl)
    
                            lm1_l.append(lm1.detach().cpu().item())
                            lm2_l.append(lm2.detach().cpu().item())
                            lm3_l.append(loss_quadrupletas.detach().cpu().item())
                            lm1_l_b.append(lm1.detach().cpu().item())
                            lm2_l_b.append(lm2.detach().cpu().item())
                            lm3_l_b.append(loss_quadrupletas.detach().cpu().item())
    
                            trlosses.append(loss_quadrupletas.detach().cpu().item())
                            trlosses_b.append(loss_quadrupletas.detach().cpu().item())
    
                            if contador_molecula%corte == 0:
                                lm1 /= (len(dataset)/corte)
                                lm2 /= (len(dataset)/corte)
                                loss_quadrupletas /= (len(dataset)/corte)
                            else:
                                # print(contador_molecula%corte)
                                lm1 /= (len(dataset)/(contador_molecula%corte))
                                lm2 /= (len(dataset)/(contador_molecula%corte))
                                loss_quadrupletas /= (len(dataset)/(contador_molecula%corte))
                           
    
                            # optimizer_y.zero_grad()
                            # optimizer_z.zero_grad()
                            # starto= datetime.now()
                            lm1.backward(retain_graph=True)  # Retener el grafo para la siguiente retropropagación
                            lm2.backward(retain_graph=True)
                            loss_quadrupletas.backward()
    
                            
                            # optimizer_y.step()
                            # optimizer_z.step()
                            
    
                            
                            # finalo =  datetime.now()
                            # print(f'dif optimizer {finalo-starto}')
    
                            # limpia listas 
                            quadrupletes_changed, quadruple_probs_list, train_noise_edge_b_list, minib_nls = [], [], [], []
                            score_des_l, score_haz_l = [],[]
                except Exception as e:
                    executor.shutdown(wait=True)  # Shut down the broken executor
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=24)
                    print(f"\033[31m {e} \033[0m")
                    continue
                count=count+1
                # print("=======")
                # limpia listas de molecula
                final = datetime.now()
                # print("dif modelo y losses:",final-start ) 
            
            start = datetime.now()
            batch +=1
            batch_maximos +=1

            # printeo pesos 
            # for name,param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad)
            try:
                optimizer_pretrained.step()
                optimizer_new.step()
                optimizer_pretrained.zero_grad()
                optimizer_new.zero_grad()
            except Exception as e:
                print(f"\033[31m {e} \033[0m")
                continue
            

            # final = datetime.now()
            # print("dif optimizer:",final-start ) 

            
            
            finalt = datetime.now()
            print("dif total:",finalt-startt )
            print("loss q",np.mean(lm3_l_b))
            print("lm1",np.mean(lm1_l_b))
            print("lm2",np.mean(lm2_l_b))
            print("bytes ", sumabytes)

            # s2 = tracemalloc.take_snapshot() 

            # top_stats = s2.compare_to(s1, 'lineno')
            # for stat in top_stats[:20]:
            #     print(stat)
        
            # print("loss_accum_b", np.mean(trlosses_b))
            
            trlosses_b_l.append(np.mean(trlosses_b))
            trlosses_b = []
            
            lm1_l_b_l.append(np.mean(lm1_l_b))
            lm1_l_b = []
            lm2_l_b_l.append(np.mean(lm2_l_b))
            lm2_l_b = [] 
            lm3_l_b_l.append(np.mean(lm3_l_b))
            lm3_l_b = [] 

            # torch.cuda.empty_cache()


            if (batch_maximos % 5) == 0: # podria hacer que esto salte cada mas o cuando se acumulen deepcopys intermedios
                start = datetime.now()
                gc.collect()
                final = datetime.now()
                print(f"gc {final-start}")

            

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

                mean_train_loss = np.mean(trlosses)
                mean_train_loss1 = np.mean(lm1_l)
                mean_train_loss2 = np.mean(lm2_l)
                mean_train_loss3 = np.mean(lm3_l)
                mean_train_losses.append(mean_train_loss)
                mean_train_losses1.append(mean_train_loss1)
                mean_train_losses2.append(mean_train_loss2)
                mean_train_losses3.append(mean_train_loss3)
                print("epoch_loss",mean_train_loss)
                
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
                
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_pretrained_state_dict': optimizer_pretrained.state_dict(),
                    'optimizer_new_state_dict': optimizer_new.state_dict(),
                    'scheduler_pretrained_state_dict': scheduler_pretrained.state_dict(),
                    'scheduler_new_state_dict': scheduler_new.state_dict(),
                    'loss': loss_quadrupletas}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}.pth')
        # scheduler_y.step(epoch)
        # scheduler_z.step(epoch)
        mean_train_loss = np.mean(trlosses)
        mean_train_loss1 = np.mean(lm1_l)
        mean_train_loss2 = np.mean(lm2_l)
        mean_train_loss3 = np.mean(lm3_l)
        mean_train_losses.append(mean_train_loss)
        mean_train_losses1.append(mean_train_loss1)
        mean_train_losses2.append(mean_train_loss2)
        mean_train_losses3.append(mean_train_loss3)
        print("epoch_loss",mean_train_loss)
        
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

import pickle
from torch.utils.data import ConcatDataset, DataLoader

import gc
import argparse

        
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

    
    with open('Data/TotalSmilesTogether.pickle', 'rb') as inf:
        df = load(inf)
        df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

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
    
    # Obtener la fecha actual
    current_date = datetime.now() 

    # Convertir la fecha al formato deseado
    # date_str = current_date.strftime('%y%m%d')  + "only_quads"
    date_str = "Prueba_Finetune_ffnet_fps"
    
    # Crear directorios si no existen
    files_dir = os.path.join("files", date_str)
    models_dir = os.path.join("models", date_str)
    resultados_dir = os.path.join(f"files/{date_str}/resultados", date_str)

    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
    
    
    #model = GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2_morgan() 
    
   
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
        
    #model = model.to(device)

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
        main(train_dl, test_dl, finetune_model, checkpoint, executor, slice=args.slice, epoch=args.epoch,
             optimizers=(optimizer_pretrained, optimizer_new),
             schedulers=(scheduler_pretrained, scheduler_new))