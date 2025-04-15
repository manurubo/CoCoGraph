# Dependency Documentation

This document outlines the function dependencies across the project files.

## Main Sender Files

### main_sender_timepred.py
- `run_script(start_index, num_molecules, subproceso, epoca)`
  - Calls: `subprocess.run()` with `main_time_pred.py`
- `main()`
  - Uses: `pickle.load()`
  - Calls: `run_script()`

### main_sender_mainmodel.py
- `run_script(start_index, num_molecules, subproceso, epoca)`
  - Calls: `subprocess.run()` with `main_single_fast_v11_ffnet.py`
- `main()`
  - Uses: `pickle.load()`
  - Calls: `run_script()`

### main_sender_mainmodel_fps.py
- `run_script(start_index, num_molecules, subproceso, epoca)`
  - Calls: `subprocess.run()` with `main_single_fast_v11_ffnet_fps_finetune.py`
- `main()`
  - Uses: `pickle.load()`
  - Calls: `run_script()`

### main_sender_timepred_fps_finetune.py
- `run_script(start_index, num_molecules, subproceso, epoca)`
  - Calls: `subprocess.run()` with `main_time_pred_fps_finetune.py`
- `main()`
  - Uses: `pickle.load()`
  - Calls: `run_script()`

## Main Implementation Files

### main_single_fast_v11_ffnet.py
- `calculate_global_probabilities_vectorized(quadruplet_probabilities, final_mask_q)`
  - Uses: Tensor operations
- `quadruplet_probability_task(args)`
  - Processes: Single quadruplet probability calculation
- `calculate_quadruplet_probabilities(pairs_break, pairs_make, adjacency_matrix)`
  - Uses: Tensor operations, itertools
  - Calls: `quadruplet_probability_task()`
- `generate_swap_tensors(final_swaps, num_nodes=MAX_ATOM)`
  - Creates: Tensors representing swaps
- `generate_swap_tensors_optimized(final_swaps, num_nodes=MAX_ATOM, device=device)`
  - Optimized version of `generate_swap_tensors()`
- `genera_intermedio(graph, swaps_to_undo)`
  - Manipulates: NetworkX graph
- `compute_features(graph, num, swaps_to_undo)`
  - Calls:
    - `genera_intermedio()`
    - `embed_edges_manuel()` from `data_preparation_utils`
    - `embed_graph_nodes_norm()` from `data_preparation_utils`
    - `embed_edges_with_cycle_sizes_norm()` from `data_preparation_utils`
    - `calculate_2d_distances_ordered()` from `data_preparation_utils`
- `save_plot_data(data, filename)`
  - Uses: `json.dump()`
- `main(train_dl, test_dl, model, checkpoint, executor, slice, epoch)`
  - Sets up optimizers
  - Core training/inference logic

### main_time_pred.py
- `genera_intermedio(graph, swaps_to_undo)`
  - Manipulates graph by swapping edges
- `compute_features(graph, num, swaps_to_undo)`
  - Calls:
    - `genera_intermedio()`
    - `embed_edges_manuel()`
    - `embed_graph_nodes_norm()`
    - `embed_edges_with_cycle_sizes_norm()`
    - `calculate_2d_distances_ordered()`
- `compute_features_cero(grafo_i)`
  - Similar to `compute_features()` but without transformations
- `compute_features_timepred(graph, num, swaps_to_undo)`
  - Calls:
    - `genera_intermedio()`
    - `embed_graph_nodes_norm_timepred()`
- `save_plot_data(data, filename)`
  - Saves plot data to JSON
- `main(train_dl, test_dl, model, checkpoint, executor, slice, epoch, saca_grafo, inversa)`
  - Sets up optimizer and scheduler
  - Handles training and validation loops
  - Uses: `GINETimePredictor` from `models.py`
  - Uses `connected_double_edge_swap()` from `adjacency_utils`

### sample-fast-molecularformula-multimolecule_compartefm.py
- `sample_positions(cumulative_distribution, shape)`
  - Uses: Tensor operations
  - Returns a position according to probability distribution
- `sample_step(current_graph_molecule, tensor, probs_quadrupletas_mod, all_smiles_molecule, idp, num_swaps, contador_molecula)`
  - Calls:
    - `components_to_graph()` from `adjacency_utils`
    - `nx_to_rdkit()` from `adjacency_utils`
    - `Chem.MolToSmiles()` from RDKit
  - Performs edge swap sampling
- `sample_step_graph(initial_graph, tensor, probs_quadrupletas_mod, all_smiles_molecule, idp, num_swaps, contador_molecula)`
  - Calls: `components_to_graph()` from `adjacency_utils`
  - Wrapper for graph-based sampling
- `apply_swap_and_count_cycles_g(graph, i1, j1, i2, j2)`
  - Calls: `count_cycles_by_size()` from `data_preparation_utils`
  - Performs edge swap and counts resulting cycles
- `calculate_data_molecule(graph, tensor, num_swaps, current_swap)`
  - Calls:
    - `components_to_graph()` from `adjacency_utils`
    - `nx_to_rdkit()` from `adjacency_utils`
    - `embed_graph_nodes_norm()` from `data_preparation_utils`
    - `embed_edges_with_cycle_sizes_norm()` from `data_preparation_utils`
    - `calculate_2d_distances_ordered()` from `data_preparation_utils`
  - Processes molecule data for inference
- `process_batch(conjunto, model, num, b_molecule, cantidad, time_model)`
  - Uses models to predict and generate molecules
  - Calls:
    - `connected_double_edge_swap()` from `adjacency_utils`
    - `embed_edges_manuel()` from `data_preparation_utils`
    - `calculate_data_molecule()`

## Utility Functions

### data_preparation_utils.py
- `embed_edges_with_cycle_sizes_norm(graph)`
  - Embeds edge information with cycle features
- `embed_edges_manuel(graph, node_list)`
  - Creates edge embeddings
- `embed_graph_nodes_norm(graph)`
  - Embeds node information
- `embed_graph_nodes_norm_timepred(graph)`
  - Specialized version for time prediction
- `calculate_2d_distances_ordered(graph, node_list)`
  - Calculates 2D distances between nodes
- `smiles_to_graph(smiles)`
  - Converts SMILES string to graph representation
- `count_cycles_by_size(graph)`
  - Counts cycles by size in the graph

### adjacency_utils.py
- `generate_padding_mask(adjacency_matrix, target)`
  - Creates mask for adjacency matrix
- `connected_double_edge_swap(G, nswap=1, _window_threshold=3)`
  - Performs edge swaps in graph
  - Used for noise generation and transformation
- `components_to_graph(nodes, adjacency_matrix)`
  - Builds graph from components and adjacency matrix
- `nx_to_rdkit(G, kekulize=True)`
  - Converts NetworkX graph to RDKit molecule

### data_loader.py
- `build_dataset_alejandro(...)`
  - Builds PyTorch Geometric dataset
  - Used for loading molecular data

### losses.py
- `loss_func_vs_inicio(...)`
  - Custom loss function for model training

## Model Classes

### models.py
- `GATN` - Base Graph Attention Network
- `GINEdgeQuadrupletPredictor` - Enhanced GNN model
  - Forward method calls multiple internal submodules:
    - GIN layers
    - Linear transformations
    - Pooling operations
- `GINETimePredictor` - Time prediction GNN model
  - Used in `main_time_pred.py`
  - Predicts diffusion time step

## Core Dependencies

### libraries.py
Imports:
- Standard Python: `sys`, `random`, `pickle`, `numpy`, `pandas`, `networkx`, `matplotlib`, `seaborn`, `rdkit`, `math`, `tqdm`, `datetime`
- PyTorch: `torch`, `torch.nn`, `torch_geometric.data`, `torch_geometric.loader`, `torch.nn.functional`, `torch.nn.Linear`, `torch_geometric.nn`, `torch.distributions`, `torch.optim`

### config.py
Defines constants:
- `NSTEP`, `ENCEL`, `FTR`, `FVA`, `seed`, `MAX_ATOM`
- `NNFEAT`, `NGFEAT`, `NHEAD_MOLFORMER`, `NHEAD`, `NGFEAT_EXTRA`, `NNFEAT_EXTRA`
- `DESCRIPTORES`, `GDESCRIPTORES`

## Workflow Summary

1. The sender files (`main_sender_*.py`) serve as entry points, launching the appropriate main implementation with specific parameters via subprocess.

2. The implementation files contain model instantiation, data loading, and training/inference logic:
   - `main_single_fast_v11_ffnet.py` - For the main GNN model
   - `main_time_pred.py` - For the time prediction model
   - `sample-fast-molecularformula-multimolecule_compartefm.py` - For molecule generation/sampling

3. All implementations rely on utility functions from:
   - `data_preparation_utils.py` - For graph/node/edge feature embedding
   - `adjacency_utils.py` - For graph manipulation
   - `data_loader.py` - For dataset construction
   - `losses.py` - For custom loss functions

4. The core models are defined in `models.py`, with the primary ones being:
   - `GINEdgeQuadrupletPredictor` - Main generative model
   - `GINETimePredictor` - Time prediction model

5. Configuration constants are defined in `config.py` and imported through `libraries.py`. 