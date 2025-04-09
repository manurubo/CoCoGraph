# CoCoGraph: A Collaborative Constrained Graph Diffusion Model

This is the official GitHub repository for the paper: **CoCoGraph: A collaborative constrained graph diffusion model for the generation of realistic synthetic molecules**.

## Overview

CoCoGraph introduces a novel approach to molecular generation using a collaborative constrained discrete diffusion model. Our model incorporates two key innovations:

1. **Valence Constraint**: A discrete double edge-swapping (DES) process that ensures each atom maintains correct valence throughout the diffusion trajectory. By building chemical constraints directly into the process, our model doesn't need to learn basic chemistry rules, allowing for significantly fewer parameters while focusing on learning what makes molecules realistic.

2. **Collaborative Mechanism**: Two models work together—a diffusion model that predicts DES operations during denoising and a time model that guides the denoising process. The time model estimates how close a molecular graph is to a valid molecule, helping the diffusion model adjust its predictions based on actual progress.

CoCoGraph achieves 100% chemical validity in generated molecules and significantly outperforms state-of-the-art approaches on the Guacamol benchmark while requiring an order of magnitude fewer parameters.

## Environment Setup

To run CoCoGraph, you need to create a Python environment with the required dependencies:

```bash
# Create a new conda environment
conda create -n cocograph python=3.9
conda activate cocograph

# Install dependencies using the requirements file
pip install -r requirements_vast.txt
```

## Directory Structure

The repository is organized as follows:

- **Data/**: Contains datasets used in the scripts
  - Molecular databases in pickle format
  - Molecular formulas for generation

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
  - `sample-fast-molecularformula-multimolecule_*.py`: Scripts for molecule generation

- **compare_guacamol/**: Used to compare results and generate graphs for the paper
  - Benchmarking scripts against other models (JTVAE, DiGress)
  - Visualization scripts for molecular properties

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
           --input_smiles_csv Data/molecular_formulas.csv \
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
           --input_smiles_csv Data/molecular_formulas.csv \
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
-   `--input_smiles_csv`: Path to the input CSV file containing SMILES strings (default: `Data/molecular_formulas.csv`).
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

1. **Property Distribution Analysis**:
   ```bash
   python compare_guacamol/compare4_composite.py
   ```
   This script compares distributions of molecular properties between generated and real molecules.

2. **Benchmark Evaluation**:
   ```bash
   python compare_guacamol/compare4_guacamol_composite.py
   ```
   This script evaluates generated molecules against the Guacamol benchmark, comparing with JTVAE and DiGress.

The results include:
- Validity, uniqueness, and novelty metrics
- Jensen-Shannon distances for property distributions
- Visualization of molecular property distributions

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

[Add license information here] 