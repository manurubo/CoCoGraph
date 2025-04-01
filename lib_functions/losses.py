# from lib_functions.config import *

# import torch 

from lib_functions.libraries import *
from lib_functions.config import *
from lib_functions.adjacency_utils import round_half_up
import torch 

def loss_func_vs_inicio(score_des, score_haz, real, entradas, mask, nls):
    BCE = torch.nn.BCELoss(reduction='none')
    probs_des = torch.sigmoid(score_des)
    probs_haz = torch.sigmoid(score_haz)

    probs_masked_des = probs_des*mask.to(device)
    probs_masked_haz = probs_haz*mask.to(device)
    
    difference_matrix_deshace = (entradas > real).float()
    difference_matrix_hace = (entradas < real).float()
    

    probs_masked_des = probs_masked_des*(entradas.to(device)>0.5).int()
    mask_no_ceros = probs_masked_des != 0
    probs_masked_no_ceros = torch.where(mask_no_ceros, probs_masked_des, torch.tensor(float('inf')))

    
    loss_matrix1 = BCE(probs_masked_des,difference_matrix_deshace.to(device))
    total1 = difference_matrix_deshace.sum(dim=(1,2))
    total1 = torch.clamp(total1, min=1)
    mask_masked = mask *(entradas>0.5).int()
    totalm = mask_masked.sum(dim=(1,2))

    enlaces_des = totalm
    
    total0 = totalm - total1
    total0 = torch.clamp(total0, min=1)

    # Calcular los pesos para cada clase
    peso_clase_0 = totalm / (2 * total0)
    peso_clase_1 = totalm / (2 * total1)
    
    weight_matrix = difference_matrix_deshace * (peso_clase_1.unsqueeze(1).unsqueeze(2)-peso_clase_0.unsqueeze(1).unsqueeze(2)) + peso_clase_0.unsqueeze(1).unsqueeze(2)

    loss_matrix1 = loss_matrix1 * weight_matrix.to(device)

    probs_masked_haz = probs_masked_haz*(entradas.to(device)<2.5).int() 
    mask_no_ceros = probs_masked_haz != 0
    probs_masked_no_ceros = torch.where(mask_no_ceros, probs_masked_haz, torch.tensor(float('inf')))
    
    loss_matrix2 = BCE(probs_masked_haz,difference_matrix_hace.to(device))
    total1 = difference_matrix_hace.sum(dim=(1,2))
    total1 = torch.clamp(total1, min=1)
    mask_masked = mask *(entradas<2.5).int()
    totalm = mask_masked.sum(dim=(1,2))

    enlaces_haz = totalm
    
    total0 = totalm - total1
    total0 = torch.clamp(total0, min=1)
    # Calcular los pesos para cada clase
    peso_clase_0 = totalm / (2 * total0)
    peso_clase_1 = totalm / (2 * total1)
    
    weight_matrix2 = difference_matrix_hace * (peso_clase_1.unsqueeze(1).unsqueeze(2)-peso_clase_0.unsqueeze(1).unsqueeze(2)) + peso_clase_0.unsqueeze(1).unsqueeze(2)
    
    loss_matrix2 = loss_matrix2 * weight_matrix2.to(device)

    beta_prime = enlaces_haz / enlaces_des

    beta_prime = beta_prime.view(beta_prime.shape[0],1,1)

    loss_matrix1 = loss_matrix1 * beta_prime.to(device)
    
    return loss_matrix1.mean(), loss_matrix2.mean()


