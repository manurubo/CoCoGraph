from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2
from lib_functions.losses import loss_func_vs_inicio
from lib_functions.data_preparation_utils import embed_edges_with_cycle_sizes_norm, embed_edges_manuel
from lib_functions.data_preparation_utils import  calculate_2d_distances_ordered, embed_graph_nodes_norm
from lib_functions.adjacency_utils import generate_mask2

from lib_functions.data_loader import build_dataset_alejandro

import random 

import os

from lib_functions.adjacency_utils import connected_double_edge_swap

from copy import deepcopy

from multiprocessing import Pool

import itertools

import concurrent.futures

import json

import gc 



def calculate_global_probabilities_vectorized(quadruplet_probabilities, final_mask_q):
    
    # Sum over the last two dimensions to aggregate probabilities for each pair (i, j)
    global_break_probabilities = quadruplet_probabilities.sum(dim=[1, 2])
    global_make_probabilities = quadruplet_probabilities.sum(dim=[1, 3])
    counts_break = final_mask_q.sum([1,2])
    counts_make = final_mask_q.sum([1,3])
    counts_break[counts_break == 0] = 1 # para impedir que sea infinito, pero realmente nunca sería
    counts_make[counts_make == 0] = 1 
    
    global_break_probabilities =  global_break_probabilities/counts_break
    global_make_probabilities =  global_make_probabilities/counts_make

    

    return global_break_probabilities, global_make_probabilities

def quadruplet_probability_task(args):
    i, j, k, l, pairs_break, pairs_make, masked_pairs_break = args
    if i != k and i != l and j != k and k != l:
        P_remove_ij = masked_pairs_break[i, j].item()
        P_remove_kl = masked_pairs_break[k, l].item()
        P_add_ik = pairs_make[i, k].item()
        P_add_jl = pairs_make[j, l].item()

        return i, j, k, l, P_remove_ij * P_remove_kl * P_add_ik * P_add_jl
    return None

def calculate_quadruplet_probabilities(pairs_break, pairs_make, adjacency_matrix):
    num_nodes = MAX_ATOM  # Assuming a graph with 35 nodes
    quadruplet_probabilities = torch.zeros((num_nodes, num_nodes, num_nodes, num_nodes))

    
    pairs_break = torch.sigmoid(pairs_break.squeeze(0))
    pairs_make = torch.sigmoid(pairs_make.squeeze(0))

    pairs_break = pairs_break.detach().cpu()
    pairs_make = pairs_make.detach().cpu()
    # Create a mask where adjacency_matrix is not 0
    # adjacency_matrix = torch.tensor(adjacency_matrix)
    mask = (adjacency_matrix != 0).cpu()

    # Apply the mask to pairs_break
    masked_pairs_break = pairs_break * mask.float()

    # Loop only over edges that exist in the graph
    start2=  datetime.now()
    enlaces_des = torch.nonzero(masked_pairs_break).tolist()
    tasks = [(i, j, k, l, pairs_break, pairs_make, masked_pairs_break) 
             for (i, j), (k, l) in itertools.product(enlaces_des, repeat=2)]

    results = pool.map(quadruplet_probability_task, tasks)

    

    for result in results:
        if result:
            i, j, k, l, probability = result
            quadruplet_probabilities[i, j, k, l] = probability

    
    return quadruplet_probabilities

def generate_swap_tensors(final_swaps, num_nodes=MAX_ATOM):
    swap_tensors = []

    for swap in final_swaps:
        # Initialize an empty tensor for this swap
        swap_tensor = torch.zeros((num_nodes, num_nodes, num_nodes, num_nodes))

        # Unpack the swap indices
        u, v, x, y = swap

        # Set the corresponding indices to 1
        swap_tensor[u, x, v, y] = 1
        swap_tensor[x, u, y, v] = 1
        swap_tensor[v, y, u, x] = 1
        swap_tensor[y, v, x, u] = 1

        # Add the tensor to the list
        swap_tensors.append(swap_tensor)

    return swap_tensors

def generate_swap_tensors_optimized(final_swaps, num_nodes=MAX_ATOM, device=device):
    # Preallocate tensor on the GPU (if using GPU)
    all_swaps_tensor = torch.zeros((len(final_swaps), num_nodes, num_nodes, num_nodes, num_nodes), device=device)

    for idx, swap in enumerate(final_swaps):
        u, v, x, y = swap

        # Perform operations directly in the preallocated tensor
        all_swaps_tensor[idx, u, x, v, y] = 1
        all_swaps_tensor[idx, x, u, y, v] = 1
        all_swaps_tensor[idx, v, y, u, x] = 1
        all_swaps_tensor[idx, y, v, x, u] = 1

    # Since all operations were performed on GPU, no need for additional .to(device)
    return all_swaps_tensor

def genera_intermedio(graph, deshacer_l):
    dk = [n for n, d in graph.degree()]
    for d in deshacer_l:
        # print("xxxxxxxx")
        u = dk[d[0][0]]
        v = dk[d[0][1]]
        x = dk[d[1][0]]
        y = dk[d[1][1]]
        graph.remove_edge(u, v)
        graph.remove_edge(x, y)
        graph.add_edge(u, x)
        graph.add_edge(v, y)
    return graph
        # u = dk[d[
def compute_features(graph, num, deshacer_l):

    grafo_i = genera_intermedio(graph,deshacer_l)
    # print("1")
    ruido, _, natoms = embed_edges_manuel(grafo_i, list(grafo_i.nodes()))
    # print("2")
    gemb, nemb, distances = embed_graph_nodes_norm(grafo_i)
    # print("3")
    edge_index, edge_attr = embed_edges_with_cycle_sizes_norm(grafo_i)
    # print(edge_index, edge_attr)
    # print("4")
    dosd_positions = calculate_2d_distances_ordered(grafo_i, list(grafo_i.nodes())) # se deberia de añadir al distances
    # print(dosd_positions)
    del(grafo_i)
#     feats_add = embed_edges_with_all_features_add(graph)
    
#     print("5")
#     feats_rem = embed_edges_with_all_features_remove(graph)
#     print("6")
    return ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd_positions

# Define a function to save plot data to a JSON file
def save_plot_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def main(train_dl, test_dl, model, checkpoint, executor, slice, epoch ):


    # Parámetros comunes a ambas partes
    common_params = [p for name, p in model.named_parameters() if not name.startswith('ff_break') 
                    and not name.startswith('ff_make')
                    and not name.startswith('reducer_make') 
                    and not name.startswith('reducer_break')]
    
#     [print(name) for name, p in model.named_parameters()]

    # Parámetros para la "parte_y"
    params_y = list(model.ff_break.parameters()) + list(model.reducer_break.parameters()) + common_params

    # Parámetros para la "parte_z"
    params_z = list(model.ff_make.parameters()) + list(model.reducer_make.parameters()) + common_params


    # Crear optimizadores separados
    optimizer_y = torch.optim.Adam(params_y, lr=1e-3)
    optimizer_z = torch.optim.Adam(params_z, lr=1e-3)
    scheduler_y = torch.optim.lr_scheduler.ExponentialLR(optimizer_y, gamma = 0.995)
    scheduler_z = torch.optim.lr_scheduler.ExponentialLR(optimizer_z, gamma = 0.995)



    # # Crear optimizadores separados
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)
    
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer_y = torch.optim.Adam(params_y, lr=1e-3)
        optimizer_y.load_state_dict(checkpoint['optimizer_y_state_dict'])
        optimizer_z = torch.optim.Adam(params_y, lr=1e-3)
        optimizer_z.load_state_dict(checkpoint['optimizer_z_state_dict'])
        scheduler_y = torch.optim.lr_scheduler.ExponentialLR(optimizer_y, gamma = 0.995)
        scheduler_z = torch.optim.lr_scheduler.ExponentialLR(optimizer_z, gamma = 0.995)
        scheduler_y.load_state_dict(checkpoint['scheduler_y_state_dict'])
        scheduler_z.load_state_dict(checkpoint['scheduler_z_state_dict'])
        # epoch = checkpoint['epoch']+1

    # Print optimizer's state_dict #hacer esto en un modelo de 0 y en uno cargado que haya modificado el scheduler
    print("Optimizer's state_dict:")
    for var_name in optimizer_y.state_dict():
        print(var_name, "\t", optimizer_y.state_dict()[var_name])

    # Print scheduler's state_dict
    print("Scheduler's state_dict:")
    print(scheduler_y.state_dict())

    scheduler_y.step(scheduler_y.state_dict()['_step_count']-1)
    scheduler_z.step(scheduler_y.state_dict()['_step_count']-1)
    # To see the current learning rate:
    for param_group in optimizer_y.param_groups:
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
                if end == True: 
                    count += 1
                    continue
                

                matrices_int = []

                start = datetime.now()
                # Submit tasks
                futures = [executor.submit(compute_features, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]
                final = datetime.now()

                largo_grafos = numswaps

                # Collect results
                try:
                    results = []
                    start = datetime.now()
                    for future in futures:
                        results.append(future.result(timeout=20))
                    final = datetime.now()
                    
                    # Process results
                    start = datetime.now()
                    for result, contador_molecula in zip(results, range(len(results))):
                        ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd = result
                        # grafos_list.append(grafo_i)
                        matrices_int.append(ruido)
                        distances = torch.Tensor(distances)
                        nl = torch.tensor(sigma_i/largo_grafos * (contador_molecula + 1))
                        nls.append(nl)
                        dosd = torch.Tensor(dosd)
                        dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=nl, distances=distances, final_entrada = ruido, dosd_distances = dosd))

                    final = datetime.now()
                    
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
    
                            quadruple_probs = torch.stack(quadruple_probs_list, dim=0)
    
                            startq = datetime.now()
                            startq2 = datetime.now()
    
                            quadruple_tensors = generate_swap_tensors_optimized(quadrupletes_changed, num_nodes = MAX_ATOM, device = device)
                            
                            lista_de_tensores = [matrix.unsqueeze(0) for matrix in train_noise_edge_b_list] 
                            final_entrada = torch.cat(lista_de_tensores, dim = 0)
                            final_entrada = final_entrada.to(device)
                            final_entrada = (final_entrada>0.5).int()
                            
                            final_entrada1 = final_entrada.unsqueeze(-1).unsqueeze(-1)
                            final_entrada2 = final_entrada.unsqueeze(1).unsqueeze(2)
                            
                            final_entrada = final_entrada1 * final_entrada2
                            
                            masks_b = generate_mask2(train_mask_b[count]) # funciona al indexar?
    
                            final_mask = masks_b.repeat(len(train_noise_edge_b_list), 1, 1)
                            
                            final_mask = final_mask.to(device)
                
                            final_mask1 = final_mask.unsqueeze(-1).unsqueeze(-1)
                            final_mask2 = final_mask.unsqueeze(1).unsqueeze(2)
                            final_mask = final_mask1*final_mask2
                            
                            final_mask = final_mask * final_entrada
    
                            quadruple_probs = quadruple_probs * final_mask.to(device)
                            quadruple_tensors = quadruple_tensors * final_mask
                            
    
                            # Assuming quadruple_probs and quadruple_tensors are your tensors
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
    
                            final_mask = masks_b.repeat(len(train_noise_edge_b_list), 1, 1)
                            final_obj = train_edge_b[count].repeat(len(train_noise_edge_b_list), 1, 1)
                            single_graph_matrices = [train_edge_b[count]]
                            for ne in train_noise_edge_b_list[:-1]:
                                single_graph_matrices.append(ne)
                        
                            lista_de_tensores = [matrix.unsqueeze(0) for matrix in single_graph_matrices] # esto ya no funciona exactamente asi, verdad?
                            all_graphs_tensor = torch.cat(lista_de_tensores, dim = 0)
    
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
                           
    
                            lm1.backward(retain_graph=True)  # Retener el grafo para la siguiente retropropagación
                            lm2.backward(retain_graph=True)
                            loss_quadrupletas.backward()
    
                            # limpia listas 
                            quadrupletes_changed, quadruple_probs_list, train_noise_edge_b_list, minib_nls = [], [], [], []
                            score_des_l, score_haz_l = [],[]
                except Exception as e:
                    executor.shutdown(wait=True)  # Shut down the broken executor
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=24)
                    print(f"\033[31m {e} \033[0m")
                    continue
                count=count+1
                # limpia listas de molecula
                final = datetime.now()
            
            start = datetime.now()
            batch +=1
            batch_maximos +=1

            try:
                optimizer_y.step()
                optimizer_z.step()
                optimizer_y.zero_grad()
                optimizer_z.zero_grad()
            except Exception as e:
                print(f"\033[31m {e} \033[0m")
                continue
            

            
            finalt = datetime.now()
            print("dif total:",finalt-startt )
            print("loss q",np.mean(lm3_l_b))
            print("lm1",np.mean(lm1_l_b))
            print("lm2",np.mean(lm2_l_b))
            print("bytes ", sumabytes)

            trlosses_b_l.append(np.mean(trlosses_b))
            trlosses_b = []
            
            lm1_l_b_l.append(np.mean(lm1_l_b))
            lm1_l_b = []
            lm2_l_b_l.append(np.mean(lm2_l_b))
            lm2_l_b = [] 
            lm3_l_b_l.append(np.mean(lm3_l_b))
            lm3_l_b = [] 

            if (batch_maximos % 5) == 0: # podria hacer que esto salte cada mas o cuando se acumulen deepcopys intermedios
                start = datetime.now()
                gc.collect()
                final = datetime.now()
                print(f"gc {final-start}")

            if (batch_maximos % 1000) == 0:
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_y_state_dict': optimizer_y.state_dict(),
                    'optimizer_z_state_dict': optimizer_z.state_dict(),
                    'scheduler_y_state_dict': scheduler_y.state_dict(),
                    'scheduler_z_state_dict': scheduler_z.state_dict(),
                    'loss': loss_quadrupletas}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.pth')
                scheduler_y.step()
                scheduler_z.step()
                # To see the current learning rate:
                for param_group in optimizer_y.param_groups:
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
                    'optimizer_y_state_dict': optimizer_y.state_dict(),
                    'optimizer_z_state_dict': optimizer_z.state_dict(),
                    'scheduler_y_state_dict': scheduler_y.state_dict(),
                    'scheduler_z_state_dict': scheduler_z.state_dict(),
                    'loss': loss_quadrupletas}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}.pth')
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
    #date_str = "240820_allmolecules_norm_largo"
    date_str = "Prueba_CoCoGraph2"
    
    # Crear directorios si no existen
    files_dir = os.path.join("files", date_str)
    models_dir = os.path.join("models", date_str)
    resultados_dir = os.path.join(f"files/{date_str}/resultados", date_str)

    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
    
    
    model = GATN_35_onlyGNNv3_quadlogits_EnhancedGIN_edges_DosD_v2() 
    
   
    if (args.epoch == 0) & (args.slice == 0):
        checkpoint = None
    else:
        if args.slice ==0:
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch-1}_slice_{22}.pth')
        else: 
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch}_slice_{args.slice-1}.pth')
        
    model = model.to(device)

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        main(train_dl, test_dl, model, checkpoint, executor, args.slice, args.epoch)