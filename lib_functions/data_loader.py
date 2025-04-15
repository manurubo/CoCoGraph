from lib_functions.config import *
from lib_functions.data_preparation_utils import embed_edges_manuel, smiles_to_graph
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset, DataLoader
from func_timeout import func_timeout


def graph_collate_fn(batch):
    """Collate function for DataLoader that handles batches of graph data.

    Args:
        batch: A list of tuples, where each tuple contains 
               (graph, tensor, smiles, atoms).

    Returns:
        A tuple containing:
        - A list of graphs.
        - A tensor stacking the input tensors.
        - A list of SMILES strings.
        - A list of atom counts.
    """
    graphs, tensors, smiles, atoms = zip(*batch)
    tensors_tensor = [torch.stack(items, dim=0) for items in zip(*tensors)]
    return graphs, tensors_tensor[0], smiles, atoms

class GraphDataset(Dataset):
    """A PyTorch Dataset for storing NetworkX graphs."""
    def __init__(self, graph_list):
        """Initializes the GraphDataset.

        Args:
            graph_list (list): A list of NetworkX graph objects.
        """
        self.graph_list = graph_list
    
    def __len__(self):
        """Returns the number of graphs in the dataset."""
        return len(self.graph_list)
    
    def __getitem__(self, idx):
        """Retrieves the graph at the specified index.

        Args:
            idx (int): The index of the graph to retrieve.

        Returns:
            networkx.Graph: The graph at the specified index.
        """
        graph = self.graph_list[idx]
        return graph
    
class PairedDataset(Dataset):
    """A PyTorch Dataset that pairs NetworkX graphs with corresponding tensors, 
    SMILES strings, and atom counts."""
    def __init__(self, graph_list, tensor_dataset, smiles_list):
        """Initializes the PairedDataset.

        Args:
            graph_list (list): A list of NetworkX graph objects.
            tensor_dataset (torch.utils.data.Dataset): A dataset containing tensors 
                                                       corresponding to the graphs. Tensors are the adjacency matrices of the graphs.
            smiles_list (list): A list of SMILES strings corresponding to the graphs.
        """
        self.graph_list = graph_list
        self.tensor_dataset = tensor_dataset
        self.smiles_list = smiles_list
        self.num_atoms_list = [graph.number_of_nodes() for graph in graph_list]  # Number of atoms in each graph
    
    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.graph_list)
    
    def __getitem__(self, idx):
        """Retrieves the graph, tensor, SMILES string, and atom count 
           at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing (graph, tensor, smiles, num_atoms).
        """
        graph = self.graph_list[idx]
        tensor = self.tensor_dataset[idx]
        smiles = self.smiles_list[idx]
        num_atoms = self.num_atoms_list[idx]  # Number of atoms
        return graph, tensor, smiles, num_atoms


def parallel_preprocessing(smiles, min_atom):
    """Processes a list of SMILES strings to generate graphs and embeddings,
    filtering by atom count and connectivity. Calculates bond statistics.

    Uses func_timeout to prevent excessively long processing for single molecules.

    Args:
        smiles (list): A list of SMILES strings.
        min_atom (int): The minimum number of atoms a molecule must have to be included.

    Returns:
        tuple: Contains lists of valid graphs, edge embeddings, SMILES strings,
               bond counts, atom counts, the total number of valid molecules,
               total bond counts across all valid molecules, and the total possible bonds.
    """
    max_atom = MAX_ATOM
    g0_list = []
    eemb_list = []
    sm_list = []
    nbonds_list = []
    natoms_list = []
    nmol = 0
    nbonds_tot = np.array([0,0,0,0])
    pos_bonds = 0
    timeout_f = 5

    for sm in tqdm(smiles):
        try:
            g0 = func_timeout(timeout_f, smiles_to_graph, args=([sm]))
        except:
            continue
        try:
            componentes = func_timeout(5, nx.number_connected_components, args=([g0]))
            if g0 != None and g0.number_of_nodes() <= max_atom and g0.number_of_nodes() >= min_atom and componentes == 1:
                nmol += 1
                node_list = list(g0.nodes())
                eemb,nbonds,natoms = func_timeout(timeout_f, embed_edges_manuel, args=(g0, node_list))
                g0_list.append(g0)
                eemb_list.append(torch.Tensor.numpy(eemb))
                sm_list.append(sm)
                nbonds_tot += np.array(nbonds)
                pos_bonds +=(natoms*(natoms-1))
        except:
            continue

    return g0_list, eemb_list, sm_list, nbonds_list, natoms_list, nmol, nbonds_tot, pos_bonds

import gc 

def build_dataset_alejandro(all_smiles, ftr=0.8, fva=0.1, bs=(NSTEP+1)*100, nsteps=NSTEP, min_atom=5, max_atom=MAX_ATOM):
    """Builds training, validation, and test datasets and dataloaders from a list
    of SMILES strings using parallel preprocessing.

    Args:
        all_smiles (list): A list of all SMILES strings for the dataset.
        ftr (float, optional): Fraction of data to use for training. Defaults to 0.8.
        fva (float, optional): Fraction of data to use for validation. Defaults to 0.1.
        bs (int, optional): Batch size for the DataLoaders. Defaults to (NSTEP+1)*100.
        nsteps (int, optional): Number of steps (used in default batch size calculation). Defaults to NSTEP.
        min_atom (int, optional): Minimum number of atoms for filtering molecules. Defaults to 5.
        max_atom (int, optional): Maximum number of atoms for filtering molecules. Defaults to MAX_ATOM.

    Returns:
        tuple: Contains the training DataLoader, validation DataLoader, test DataLoader,
               and the percentage of each bond type across the dataset.
    """
    # Prepare the whole dataset with all the embeddings
    g_list, e_list, s_list, nmol = [], [], [], 0
    pos_bonds = 0
    nbonds_tot = np.array([0,0,0,0])

    num_splits = 100
    all_smiles_split = np.array_split(all_smiles, num_splits)

    import concurrent
    futures = []
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=16)

    for smiles_sm in all_smiles_split:
        future = executor.submit(parallel_preprocessing, smiles_sm, min_atom)
        futures.append(future)


    for future in futures:
        g0, eemb, sm, nbons, natoms, nmol_a, nbonds_tot_a, pos_bonds_a = future.result()
        nmol += nmol_a
        g_list += g0
        e_list += [torch.from_numpy(x) for x in eemb]
        s_list += sm
        nbonds_tot += np.array(nbonds_tot_a)
        pos_bonds += pos_bonds_a
        del(g0, eemb, sm, nbons, natoms, nmol_a, nbonds_tot_a, pos_bonds_a)
        gc.collect()


    nbonds_perc =nbonds_tot/pos_bonds
    print(len(g_list))
    Ntr, Nva = int(nmol*ftr), int(nmol*fva)
    
    e_tensor = torch.stack(e_list, dim=0)
    del(e_list)
    gc.collect()
    
    train_gr = GraphDataset(g_list[:Ntr])
    train_e = TensorDataset(e_tensor[:Ntr])
    train_s = s_list[:Ntr]  # Train SMILES
    print(len(train_gr))
    validation_gr = GraphDataset(g_list[Ntr:Ntr+Nva])
    validation_e = TensorDataset(e_tensor[Ntr:Ntr+Nva])
    validation_s = s_list[Ntr:Ntr+Nva]  # Validation SMILES
    print(len(validation_gr))
    test_gr = GraphDataset(g_list[Ntr+Nva:])
    test_e = TensorDataset(e_tensor[Ntr+Nva:])
    test_s = s_list[Ntr+Nva:]  # Test SMILES
    print(len(test_gr))
    combined_dataset_tr = PairedDataset(train_gr, train_e, train_s)
    combined_dataset_va = PairedDataset(validation_gr, validation_e, validation_s)
    combined_dataset_te = PairedDataset(test_gr, test_e, test_s)
    
    train_dl = DataLoader(combined_dataset_tr, batch_size=bs, collate_fn=graph_collate_fn)
    validation_dl = DataLoader(combined_dataset_va, batch_size=bs, collate_fn=graph_collate_fn)
    test_dl = DataLoader(combined_dataset_te, batch_size=bs, collate_fn=graph_collate_fn)
    
    return train_dl, validation_dl, test_dl, nbonds_perc