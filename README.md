# CoCoGraph: A Collaborative Constrained Graph Diffusion Model

This is the official GitHub repository for the paper: **CoCoGraph: A collaborative constrained graph diffusion model for the generation of realistic synthetic molecules**.

## Overview

CoCoGraph introduces a novel approach to molecular generation using a collaborative constrained discrete diffusion model. Our model incorporates two key innovations:

1. **Valence Constraint**: A discrete double edge-swapping (DES) process that ensures each atom maintains correct valence throughout the diffusion trajectory. By building chemical constraints directly into the process, our model doesn't need to learn basic chemistry rules, allowing for significantly fewer parameters while focusing on learning what makes molecules realistic.

2. **Collaborative Mechanism**: Two models work together—a diffusion model that predicts DES operations during denoising and a time model that guides the denoising process. The time model estimates how close a molecular graph is to a valid molecule, helping the diffusion model adjust its predictions based on actual progress.

CoCoGraph achieves 100% chemical validity in generated molecules and significantly outperforms state-of-the-art approaches on the Guacamol benchmark while requiring an order of magnitude fewer parameters.

## System Requirements

### Software Dependencies
- **Python**: 3.9.x
- **PyTorch**: ≥1.12.0
- **RDKit**: ≥2022.03.1
- **NumPy**: ≥1.21.0
- **Pandas**: ≥1.4.0
- **Scikit-learn**: ≥1.1.0
- **Matplotlib**: ≥3.5.0
- **Seaborn**: ≥0.11.0
- **Tqdm**: ≥4.64.0

### Operating Systems Tested
- Ubuntu 20.04 LTS
- Ubuntu 22.04 LTS

### Hardware Requirements
- **Minimum**: 8 GB RAM, 4 CPU cores
- **Recommended**: 16+ GB RAM, 8+ CPU cores, GPU with 8+ GB VRAM
- **Storage**: 10+ GB free disk space for models and generated molecules

### Non-standard Hardware
- CUDA-capable GPU recommended for training (optional for inference)
- Training large models may require GPUs with 16+ GB VRAM

## Installation Guide

### Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/manurubo/CoCoGraph.git
   cd CoCoGraph
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n cocograph python=3.9
   conda activate cocograph
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_vast.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; import rdkit; print('Installation successful!')"
   ```

### Typical Install Time
- **On a normal desktop computer**: 10-15 minutes
- Note: Initial conda environment creation and package downloads may take longer on slower connections

## Demo

### Instructions to Run Demo

We provide a quick demo using a small subset of molecules to test the installation and basic functionality:

1. **Generate a small set of molecules using pre-trained models**:
   ```bash
   # Demo with BASE model (uses first 100 molecules from dataset)
   python sample_scripts/sample_molecules_BASEmodel.py \
          --input_smiles_csv Data/molecules_lt70atoms_annotated.csv \
          --output_dir_suffix demo_run \
          --model_checkpoint_path models/BASE_diffusion/model_epoch_2_slice_22.pth \
          --time_model_checkpoint_path models/BASE_time/model_epoch_2_slice_22.pth \
          --batch_size_sample 100 \
          --batch_size_process 5 \
          --save_every_n_batches 1 \
          --num_workers 2
   ```

2. **Evaluate demo results**:
   ```bash
   # Property distribution analysis (comparing generated molecules to original dataset and baselines)
   python compare_guacamol/compare4_composite.py \
          -gen -ori -jtvae -digress \
          --reference ori \
          --directory demo_run
   
   # Guacamol benchmark evaluation (if you want standard benchmarks)
   python compare_guacamol/compare4_guacamol_composite.py \
          -gen -ori -jtvae -digress \
          --reference ori \
          --directory demo_run
   ```

### Expected Output
- **Generated molecules**: 100 new molecules saved in `mols_gen/demo_run/`
- **Validity**: 100% (all generated molecules are chemically valid)
- **Log files**: Training progress and generation statistics
- **Property distributions**: Basic molecular property analysis

### Expected Runtime for Demo
- **On a normal desktop computer**: 2-5 minutes
- **With GPU acceleration**: 1-2 minutes
- **CPU only**: 5-10 minutes



## Directory Structure

The repository is organized as follows:

- **Data/**: Contains datasets used in the scripts
  - Molecular databases in pickle format
  - Molecular formulas for generation
  - **generated_database/**: Contains large synthetic molecular databases.
    - `all_molecules.csv`: A comprehensive database of 8.2 million generated molecules.
    - `novel_molecules.csv`: A subset of `all_molecules.csv` containing 7.6 million unique and novel molecules.
    *Note: These files are managed with Git LFS due to their size.*

- **lib_functions/**: Contains helper functions used throughout the codebase
  - `adjacency_utils.py`: Utilities for handling molecule adjacency matrices
  - `config.py`: Configuration parameters and settings
  - `data_loader.py`: Utilities for loading and processing data
  - `data_preparation_utils.py`: Utilities for data preparation
  - `libraries.py`: Import statements for external libraries
  - `losses.py`: Loss functions for model training
  - `models.py`: Neural network model definitions
  - `sample_utils.py`: Utilities for molecule sampling

- **main_scripts/**: Contains the main code to launch and train the algorithms
  - `main_sender_*.py`: Sender scripts to initiate training
  - `main_single_*.py`: Scripts for training the diffusion model
  - `main_time_pred*.py`: Scripts for training the time prediction model

- **sample_scripts/**: Contains scripts for generating molecules once models are trained
  - `sample_molecules_BASEmodel.py`: Script for molecule generation using BASE models (without fingerprints)
  - `sample_molecules_FPSmodel.py`: Script for molecule generation using FPS models (with fingerprints)

- **compare_guacamol/**: Used to compare results and generate graphs for the paper
  - `compare4_composite.py`: Property distribution analysis and visualization script  
  - `compare4_guacamol_composite.py`: Guacamol benchmark evaluation script
  - Benchmarking scripts against other models (JTVAE, DiGress)

- **models/**: Contains trained model weights
  - BASE models (without fingerprints)
  - FPS models (with fingerprints)

- **files/**: Contains results from training models
  - Training logs
  - Model checkpoints
  - Evaluation metrics

- **mols_gen/**: Directory that contains generated molecules
  - Molecules in SMILES format
  - Analysis results

## Training Models

All scripts must be run from the root directory of the project. The training process consists of two steps: training the diffusion model and training the time prediction model.

The sender scripts (`main_sender_*.py`) initiate the training process by launching the corresponding main training scripts in batches. These sender scripts accept several command-line arguments to control the training process, which is useful for splitting the workload across multiple processes or resuming training:

-   `--slice`: Specifies the starting slice number for the batch processing (default depends on the script, often 0). Each slice typically corresponds to processing a batch of molecules defined by `--num_molecules`.
-   `--epoch`: Specifies the starting epoch number for training (default: 0).
-   `--start_index`: Specifies the starting index within the dataset from which to begin processing molecules (default: 0).
-   `--num_molecules`: Specifies the number of molecules to process per batch (batch size for the sender script, default: 100000).

Example: To start training the BASE diffusion model from epoch 1, slice 5, processing 50,000 molecules per batch starting from molecule index 500,000:
```bash
python main_scripts/main_sender_mainmodel.py --epoch 1 --slice 5 --start_index 500000 --num_molecules 50000
```

### Diffusion Model Training

We provide two versions of the diffusion model:

1.  **BASE Model**: The core model without molecular fingerprints. Launch training using the sender script:
    ```bash
    # Example: Start training from default values (epoch 0, slice 0, index 0, 100k molecules/batch)
    python main_scripts/main_sender_mainmodel.py

    # Example: Start training from a specific point
    python main_scripts/main_sender_mainmodel.py --epoch <epoch_num> --slice <slice_num> --start_index <index> --num_molecules <batch_size>
    ```

2.  **FPS Model**: Enhanced model incorporating Morgan fingerprints as additional inputs. Launch training using the sender script:
    ```bash
    # Example: Start training from default values
    python main_scripts/main_sender_mainmodel_fps.py

    # Example: Start training from a specific point
    python main_scripts/main_sender_mainmodel_fps.py --epoch <epoch_num> --slice <slice_num> --start_index <index> --num_molecules <batch_size>
    ```

The FPS model improves edge-swapping prediction by utilizing molecular fingerprints, providing better performance at the cost of more parameters (3.1M vs 0.471M for BASE).

### Time Prediction Model Training

Similarly, there are two versions of the time prediction model:

1.  **BASE Time Model**: Launch training using the sender script:
    ```bash
    # Example: Start training from default values
    python main_scripts/main_sender_timepred.py

    # Example: Start training from a specific point
    python main_scripts/main_sender_timepred.py --epoch <epoch_num> --slice <slice_num> --start_index <index> --num_molecules <batch_size>
    ```

2.  **FPS Time Model**: Launch training using the sender script:
    ```bash
    # Example: Start training from default values
    python main_scripts/main_sender_timepred_fps_finetune.py

    # Example: Start training from a specific point
    python main_scripts/main_sender_timepred_fps_finetune.py --epoch <epoch_num> --slice <slice_num> --start_index <index> --num_molecules <batch_size>
    ```

The FPS version of the time model provides more accurate time predictions by incorporating fingerprint information, but requires more parameters (1.3M vs 0.063M for BASE).

### Configuration

Training parameters can be modified in `lib_functions/config.py`. Key parameters include:
- Batch size
- Learning rate
- Number of epochs
- Model architecture parameters
- Dataset paths

## Sampling Molecules

After training, you can generate new molecules using the sampling scripts located in the `sample_scripts/` directory. These scripts now accept command-line arguments to configure the sampling process.

1.  **With BASE models** (trained without fingerprints):
    ```bash
    python sample_scripts/sample_molecules_BASEmodel.py \
           --input_smiles_csv Data/molecules_lt70atoms_annotated.csv \
           --output_dir_suffix BASE_run_1 \
           --model_checkpoint_path models/BASE/model_epoch_X.pth \
           --time_model_checkpoint_path models/BASE_time/model_epoch_Y.pth \
           --batch_size_sample 1000 \
           --batch_size_process 50 \
           --save_every_n_batches 50 \
           --num_workers 8
    ```

2.  **With FPS models** (trained with fingerprints):
    ```bash
    python sample_scripts/sample_molecules_FPSmodel.py \
           --input_smiles_csv Data/molecules_lt70atoms_annotated.csv \
           --output_dir_suffix FPS_run_1 \
           --model_checkpoint_path models/FPS/model_epoch_X.pth \
           --time_model_checkpoint_path models/FPS_time/model_epoch_Y.pth \
           --batch_size_sample 1000 \
           --batch_size_process 50 \
           --save_every_n_batches 50 \
           --num_workers 8
    ```

These scripts use the trained diffusion and time models collaboratively to generate new molecules based on SMILES strings sampled from the input CSV.

The sampling process:
1.  Starts with random molecular graphs derived from SMILES sampled from the `--input_smiles_csv` file.
2.  Uses the diffusion model (specified by `--model_checkpoint_path`) to predict DES operations.
3.  Uses the time model (specified by `--time_model_checkpoint_path`) to guide the denoising process.
4.  Selects the molecule with the smallest predicted time as the final output for each starting SMILES.
5.  Saves results periodically and cumulatively in a directory under `mols_gen/` named with the `--output_dir_suffix`.

Key sampling parameters configurable via command-line arguments:
-   `--input_smiles_csv`: Path to the input CSV file containing SMILES strings (default: `Data/molecules_lt70atoms_annotated.csv`).
-   `--output_dir_suffix`: Suffix for the output directory where generated molecules and logs will be saved (default depends on the script).
-   `--model_checkpoint_path`: Path to the trained diffusion model checkpoint.
-   `--time_model_checkpoint_path`: Path to the trained time prediction model checkpoint.
-   `--batch_size_sample`: Number of SMILES to sample from the input file in each main loop iteration (default: 1000).
-   `--batch_size_process`: Batch size for processing molecules within the `process_batch` function (default: 50).
-   `--save_every_n_batches`: Frequency (in terms of sampling batches) for saving cumulative results (default: 50).
-   `--num_workers`: Number of worker processes for parallel execution (default: 8).

Make sure to replace `model_epoch_X.pth` and `model_epoch_Y.pth` with the actual paths to your trained model checkpoints.

## Obtaining Results

To evaluate generated molecules and reproduce the paper's results:

### Evaluation Scripts Parameters

Both comparison scripts (`compare4_composite.py` and `compare4_guacamol_composite.py`) accept the following parameters:

- `-ori`: Include original training dataset molecules in comparison
- `-gen`: Include your generated molecules in comparison  
- `-jtvae`: Include JTVAE generated molecules in comparison
- `-digress`: Include DiGress generated molecules in comparison
- `--reference` or `-ref`: Specify the reference category for comparisons (typically `ori`)
- `--directory` or `-dir`: Directory name(s) containing generated molecules (from `--output_dir_suffix`)
- `--load_descriptors`: Load pre-calculated descriptors instead of recalculating them (speeds up repeated analysis)

### Property Distribution Analysis

```bash
# Basic comparison: your model vs original dataset
python compare_guacamol/compare4_composite.py \
       -gen -ori \
       --reference ori \
       --directory your_experiment_name

# Full comparison: your model vs all baselines
python compare_guacamol/compare4_composite.py \
       -gen -ori -jtvae -digress \
       --reference ori \
       --directory your_experiment_name

# Multiple experiment comparison
python compare_guacamol/compare4_composite.py \
       -gen -ori \
       --reference ori \
       --directory experiment1 experiment2 experiment3
```

This script compares distributions of molecular properties between generated and real molecules, producing:
- Statistical comparison tables
- Jensen-Shannon distance calculations
- Property distribution plots
- Composite visualization figures

### Benchmark Evaluation (Guacamol)

```bash
# Basic Guacamol benchmark evaluation
python compare_guacamol/compare4_guacamol_composite.py \
       -gen -ori -jtvae -digress \
       --reference ori \
       --directory your_experiment_name

# Load pre-calculated descriptors for faster evaluation
python compare_guacamol/compare4_guacamol_composite.py \
       -gen -ori -jtvae -digress \
       --reference ori \
       --directory your_experiment_name \
       --load_descriptors
```

This script evaluates generated molecules against the Guacamol benchmark, producing:
- Validity, uniqueness, and novelty metrics  
- Internal similarity calculations
- Benchmark comparison with JTVAE and DiGress
- Performance ratio analysis

### Output Files

The evaluation scripts generate several output files in `mols_gen/{your_directory}/`:
- `combined_molecules_with_descriptors.csv`: All molecules with calculated descriptors
- `mytable_res_{category1}_vs_{category2}.csv`: Statistical comparison tables
- `DistBetweenDists.csv`: Jensen-Shannon distances between distributions
- `graficas_*/`: Directory containing visualization plots and figures

## Citation

If you use CoCoGraph in your research, please cite our paper:

```
@article{ruiz2023cocograph,
  title={CoCoGraph: A collaborative constrained graph diffusion model for the generation of realistic synthetic molecules},
  author={Ruiz-Botella, Manuel and Sales-Pardo, Marta and Guimerà, Roger},
  journal={},
  year={2025}
}
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by-nc-sa/4.0/)

### You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

### Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **NonCommercial** — You may not use the material for commercial purposes
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original

For more details, see the [full license text](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 