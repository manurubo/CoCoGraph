from lib_functions.libraries import *
from lib_functions.config import *

from lib_functions.models import TimePredictionModel_graph
from lib_functions.data_preparation_utils import embed_edges_with_cycle_sizes_norm, embed_edges_manuel
from lib_functions.data_preparation_utils import calculate_2d_distances_ordered, embed_graph_nodes_norm
from lib_functions.data_preparation_utils import embed_graph_nodes_norm_timepred
from lib_functions.data_preparation_utils import compute_features, compute_features_cero, compute_features_timepred, save_plot_data
from lib_functions.adjacency_utils import connected_double_edge_swap, genera_intermedio
from lib_functions.data_loader import build_dataset_alejandro

import random 
import os
from copy import deepcopy
import concurrent.futures
import json
import gc 
import argparse

def main(train_dl, test_dl, model, checkpoint, executor, slice, epoch, saca_grafo, inversa ):


    # Parámetros comunes a ambas partes
    common_params = [p for name, p in model.named_parameters() ]
    
    # Crear optimizadores 
    optimizer = torch.optim.Adam(common_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)

    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = torch.optim.Adam(common_params, lr=1e-3)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.995)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # epoch = checkpoint['epoch']+1

    # Print optimizer's state_dict #hacer esto en un modelo de 0 y en uno cargado que haya modificado el scheduler
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    # Print scheduler's state_dict
    print("Scheduler's state_dict:")
    print(scheduler.state_dict())

    scheduler.step(scheduler.state_dict()['_step_count']-1)
    # To see the current learning rate:
    for param_group in optimizer.param_groups:
        print("Current Learning Rate:", param_group['lr'])


    

    print("-------------------------------------------")

    # Initialize lists for storing mean losses
    mean_train_losses,  mean_train_losses1= [],[]
    mean_valid_losses = []
    trlosses, vlosses, trlosses_b, trlosses_b_l = [], [], [], []
    lm1_l,  lm1_l_b, lm1_l_b_l,  = [],[],[]
    batch_maximos = 0  # IMPORTANTE! ajustar por el guardado del modelo 
    epoch_max = epoch+1
    print(epoch, epoch_max)

    criterion = nn.MSELoss() # EMPIEZA EN 0.0202

    
    while epoch < epoch_max:

        dif_grados = []

        minvloss = 1e100
        trnorm = 0
        model.train()
        batch = 0
        print("epoch",epoch)

        predictions_final = []
        nls_final = []

        
        for train_graph_b, train_edge_b, smiles, atoms in train_dl:
            print("batch", batch)
            print(sum(atoms))

            startt = datetime.now()
            
            
            # aqui saca un ruido para cada ejemplo, la prob de cambio? si
            sigma_list = [0.5] * train_edge_b.size(0)
            train_mask_b = train_edge_b.sum(-1).gt(1e-3).to(dtype=torch.float32)

            count=0 
            
            sumabytes = 0
            for sigma_i in sigma_list:
                dataset = []
                nls = []
                nls_real = []
                start = datetime.now()
                num_swaps = math.ceil(sigma_i * torch.sum(train_edge_b[count]).item() / 2)
                numswaps, g_ruido, graphs_list, end, final_swaps, rem_acc , cre_acc = connected_double_edge_swap(deepcopy(train_graph_b[count]), num_swaps, seed = random.Random())

                final = datetime.now()

                if end == True: 
                    count += 1
                    continue

                matrices_int = []

                start = datetime.now()
                
                
                


                try: 

                    # Submit tasks #aqui meter el timepred o no, segun interese
                    if saca_grafo:
                        futures = [executor.submit(compute_features, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]
                    else: 
                        futures = [executor.submit(compute_features_timepred, train_graph_b[count], num, rem ) for num, rem  in enumerate(rem_acc)]

                    # results = []
                    # num_e = 0
                    # for remove, create in zip(rem_acc, cre_acc):
                    #     resultado = compute_features(deepcopy(train_graph_b[count]), num_e, remove)
                    #     results.append(resultado)
                    #     num_e += 1

                    largo_grafos = numswaps

                    # Collect results
                
                
                    results = []
                    for future in futures:
                        results.append(future.result(timeout=60))

                    # Process results
                    if saca_grafo:
                        if inversa == False:
                            nl = 0.0
                            nls.append(nl)  

                        ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd = compute_features_cero(train_graph_b[count])
                        distances = torch.Tensor(distances)
                        dosd = torch.Tensor(dosd)
                        dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=0.0, distances=distances, final_entrada = ruido, dosd_distances = dosd))                        
                        cuenta = 0 
                        for result, contador_molecula in zip(results, range(len(results))):
                            ruido, gemb, nemb, distances, edge_index, edge_attr, natoms, num, dosd = result
                            nl = torch.tensor(sigma_i/largo_grafos * (contador_molecula + 1))
                            
                                
                                
                            if inversa: 
                                if cuenta == 0:
                                    cuenta = 1
                                    nl_inicial = nl/2
                                    nl_inicial = 1/nl_inicial
                                    nls_real.append(0.0)
                                    nls.append(nl_inicial)
                                nls_real.append(nl)
                                nl = 1/nl
                            
                            nls.append(nl)

                            distances = torch.Tensor(distances)
                            dosd = torch.Tensor(dosd)
                            dataset.append(Data(x=nemb, edge_index=edge_index, xA=gemb, edge_attr=edge_attr, noiselevel=nl, distances=distances, final_entrada = ruido, dosd_distances = dosd))

                        nls = torch.Tensor(nls).to(device)
                        if inversa: nls_real = torch.Tensor(nls_real).to(device)
                        optimizer.zero_grad()

                        predictions = []
                        predictions_real = [] 
                        for graph_data in dataset:
                            graph_data = graph_data.to(device)

                            # Forward pass
                            prediction = model(graph_data)
                            if inversa: 
                                predictions_real.append(prediction)
                                prediction = 1/prediction
                            predictions.append(prediction)

                        predictions = torch.stack(predictions).squeeze(1).squeeze(1)
                        if inversa: predictions_real = torch.stack(predictions_real).squeeze(1).squeeze(1)

                    else:
                        graphs_embeddings= []
                        nls.append(0.0)
                        gemb = embed_graph_nodes_norm_timepred(train_graph_b[count])
                        graphs_embeddings.append(gemb)
                        for result, contador_molecula in zip(results, range(len(results))):
                            gemb = result
                            nl = torch.tensor(sigma_i/largo_grafos * (contador_molecula + 1))
                            nls.append(nl)
                            graphs_embeddings.append(gemb)

                        graphs_embeddings = torch.stack(graphs_embeddings).to(device)
                        nls = torch.Tensor(nls).to(device)
                        optimizer.zero_grad()

                        predictions = model(graphs_embeddings)
                        predictions = predictions.squeeze(1)  # Elimina la dimensión adicional    

    #                     print(predictions.shape,nls.shape)
                    loss = criterion(predictions, nls)
                    # para generar la grafica
                    # predictions_final.append(predictions.tolist())
                    # nls_final.append(nls.tolist())
                    
                    loss.backward()
                    optimizer.step()

                    loss_molec = loss.detach().cpu().item()
                    lm1_l_b.append(loss_molec)
                except Exception as e:
                    executor.shutdown(wait=True)  # Shut down the broken executor
                    executor = concurrent.futures.ProcessPoolExecutor(max_workers=24)
                    print(f"\033[31m {e} \033[0m")
                    continue
                
                    
            print("=============")
            batch +=1
            batch_maximos +=1
            
            loss_batch = np.mean(lm1_l_b)
            lm1_l.append(loss_batch)
            
            finalt = datetime.now()
            print("dif total:",finalt-startt )
            print("loss",loss_batch)
            
            

            if (batch_maximos % 5) == 0: # podria hacer que esto salte cada mas o cuando se acumulen deepcopys intermedios
                start = datetime.now()
                gc.collect()
                final = datetime.now()
                print(f"gc {final-start}")

                if inversa: print("usados")
                print(predictions)
                print(nls)
                if inversa: 
                    print("reales")
                    print(predictions_real)
                    print(nls_real)
                    print(criterion(predictions_real, nls_real))

            # if (batch_maximos % 5) == 0:
            #     plot_data = {
            #         'Predictions': predictions_final,
            #         'Times':nls_final
            #     }
            #     save_plot_data(plot_data, f"files/{date_str}/resultados/predictions_vs_time_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
    
            

            if (batch_maximos % 1000) == 0:
                torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.pth')
                scheduler.step()
                loss_epoch = np.mean(lm1_l)
                trlosses.append(loss_epoch)
                # To see the current learning rate:
                for param_group in optimizer.param_groups:
                    print("Current Learning Rate:", param_group['lr'])

                print("1000b_loss",loss_epoch)
                
                plot_data = {
                    'Train loss MiniB': lm1_l
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/batch_loss_timepred_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
    
                plot_data = {
                    'Train loss': trlosses,
                }
                save_plot_data(plot_data, f"files/{date_str}/resultados/epoch_loss_timepred_data_epoch_{epoch}_slice_{slice}_batch_{batch_maximos}.json")
                
                lm1_l, lm1_l_b  = [], []
                

            print("=============")
                
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss}, f'models/{date_str}/model_epoch_{epoch}_slice_{slice}.pth')
        # scheduler_y.step(epoch)
        # scheduler_z.step(epoch)
        loss_epoch = np.mean(lm1_l)
        print("epoch_loss",loss_epoch)
        trlosses.append(loss_epoch)
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
    date_str = "241017_all_timepred"
    # date_carga = "240828_all_timepred_largo_graph_continua"
    
    # Crear directorios si no existen
    files_dir = os.path.join("files", date_str)
    models_dir = os.path.join("models", date_str)
    resultados_dir = os.path.join(f"files/{date_str}/resultados", date_str)

    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(resultados_dir, exist_ok=True)
    
    saca_grafo =True
    if saca_grafo:
        model = TimePredictionModel_graph() 
    else:
        model = TimePredictionModel(input_dim=NGFEAT) 
    
   
    if (args.epoch == 0) & (args.slice == 0):
        checkpoint = None
    else:
        if args.slice ==0:
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch-1}_slice_{22}.pth')
        else: 
            checkpoint = torch.load(f'models/{date_str}/model_epoch_{args.epoch}_slice_{args.slice-1}.pth')
        
    model = model.to(device)

    with concurrent.futures.ProcessPoolExecutor(max_workers=24) as executor:
        # tracemalloc.start()
        main(train_dl, test_dl, model, checkpoint, executor, args.slice, args.epoch, saca_grafo, False)
    # main(train_dl, test_dl, model, checkpoint, None )
