import torch
# The number of steps in the graph difusion process
NSTEP = 5
# The atoms used for node/edge embedding
ENCEL = ['B', 'N', 'C', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Ca', 'K', 'Na', 'Mg', 'H']
# Fraction of samples in train and validation sets (use the remaining for testing)
FTR = 0.8
FVA = 0.1
# Random seed
seed = 1111
# Maximum number of atoms
MAX_ATOM = 70


# # Important! Calculate (by example) the number of features in the node embedding
NNFEAT = 31 
NGFEAT = 21

NHEAD_MOLFORMER = 8
NHEAD = 1

NGFEAT_EXTRA = 41
NNFEAT_EXTRA = 46

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device)


