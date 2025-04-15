from lib_functions.libraries import *
from lib_functions.config import *
from lib_functions.adjacency_utils import round_half_up
import torch 

def loss_func_vs_inicio(remove_scores, add_scores, target_adj, input_adj, mask, nls):
    """Calculates a weighted binary cross-entropy loss for two separate tasks: 
    predicting edges to remove ('deshacer') and edges to add ('hacer').

    The loss is weighted based on class imbalance within each task and also 
    adjusts the 'remove' loss based on the ratio of possible 'add' edges 
    to possible 'remove' edges.

    Args:
        remove_scores (torch.Tensor): Raw scores (logits) for the 'remove' task.
        add_scores (torch.Tensor): Raw scores (logits) for the 'add' task.
        target_adj (torch.Tensor): The target adjacency matrix (ground truth).
        input_adj (torch.Tensor): The input adjacency matrix.
        mask (torch.Tensor): A mask indicating valid positions in the matrices.
        nls: Not used in the function.

    Returns:
        tuple: 
            - torch.Tensor: Mean weighted BCE loss for the 'remove' task.
            - torch.Tensor: Mean weighted BCE loss for the 'add' task.
    """
    BCE = torch.nn.BCELoss(reduction='none')
    remove_probs = torch.sigmoid(remove_scores)
    add_probs = torch.sigmoid(add_scores)

    masked_remove_probs = remove_probs * mask.to(device)
    masked_add_probs = add_probs * mask.to(device)
    
    remove_target_matrix = (input_adj > target_adj).float()
    add_target_matrix = (input_adj < target_adj).float()
    
    # --- Remove Loss Calculation (Original Logic) ---
    masked_remove_probs = masked_remove_probs * (input_adj.to(device) > 0.5).int()
    
    loss_matrix_remove = BCE(masked_remove_probs, remove_target_matrix.to(device))
    
    # Weight calculation for remove loss
    total_target_removals = remove_target_matrix.sum(dim=(1,2))
    total_target_removals = torch.clamp(total_target_removals, min=1)
    effective_remove_mask = mask * (input_adj > 0.5).int()
    total_valid_remove_entries = effective_remove_mask.sum(dim=(1,2))

    num_possible_removals = total_valid_remove_entries # Renamed from enlaces_des
    
    total_non_target_removals = total_valid_remove_entries - total_target_removals
    total_non_target_removals = torch.clamp(total_non_target_removals, min=1)

    weight_class_0 = total_valid_remove_entries / (2 * total_non_target_removals)
    weight_class_1 = total_valid_remove_entries / (2 * total_target_removals)
    
    weight_matrix_remove = remove_target_matrix * (weight_class_1.unsqueeze(1).unsqueeze(2) - weight_class_0.unsqueeze(1).unsqueeze(2)) + weight_class_0.unsqueeze(1).unsqueeze(2)

    loss_matrix_remove = loss_matrix_remove * weight_matrix_remove.to(device)

    # --- Add Loss Calculation (Original Logic) ---
    masked_add_probs = masked_add_probs * (input_adj.to(device) < 2.5).int() 
    
    loss_matrix_add = BCE(masked_add_probs, add_target_matrix.to(device))
    
    # Weight calculation for add loss
    total_target_additions = add_target_matrix.sum(dim=(1,2))
    total_target_additions = torch.clamp(total_target_additions, min=1)
    effective_add_mask = mask * (input_adj < 2.5).int()
    total_valid_add_entries = effective_add_mask.sum(dim=(1,2))

    num_possible_additions = total_valid_add_entries # Renamed from enlaces_haz
    
    total_non_target_additions = total_valid_add_entries - total_target_additions
    total_non_target_additions = torch.clamp(total_non_target_additions, min=1)
    
    weight_class_0_add = total_valid_add_entries / (2 * total_non_target_additions)
    weight_class_1_add = total_valid_add_entries / (2 * total_target_additions)
    
    weight_matrix_add = add_target_matrix * (weight_class_1_add.unsqueeze(1).unsqueeze(2) - weight_class_0_add.unsqueeze(1).unsqueeze(2)) + weight_class_0_add.unsqueeze(1).unsqueeze(2)
    
    loss_matrix_add = loss_matrix_add * weight_matrix_add.to(device)

    # --- Final Adjustment (Original Logic) ---
    # Clamp possible removals to avoid division by zero if no removals are possible
    add_remove_ratio = num_possible_additions / torch.clamp(num_possible_removals, min=1) 

    add_remove_ratio = add_remove_ratio.view(add_remove_ratio.shape[0], 1, 1)

    # Apply ratio scaling to the remove loss
    loss_matrix_remove = loss_matrix_remove * add_remove_ratio.to(device)
    
    return loss_matrix_remove.mean(), loss_matrix_add.mean()


