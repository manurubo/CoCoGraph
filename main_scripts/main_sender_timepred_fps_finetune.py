import subprocess
import os
import sys
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib_functions.libraries import *
from lib_functions.config import *

def run_script(start_index, num_molecules, subproceso, epoca):
    # Command to run the Python script with specific arguments
    command = [
        'python', 'main_scripts/main_time_pred_fps_finetune.py', # antes era 7
        '--start_index', str(start_index),
        '--num_molecules', str(num_molecules), 
        '--slice', str(subproceso),
        '--epoch', str(epoca)
    ]
    
    # Create a new environment with PYTHONPATH set to include the current directory
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd() + (':' + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')
    
    subprocess.run(command, check=True, env=env)

def main():
    parser = argparse.ArgumentParser(description='Run time prediction FPS finetune script in batches.')
    parser.add_argument('--slice', type=int, required=True, help='Starting slice number.')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number.')
    parser.add_argument('--start_index', type=int, default=0, help='Starting molecule index.')
    parser.add_argument('--num_molecules', type=int, default=100000, help='Number of molecules per batch (batch size).')

    args = parser.parse_args()

    with open('Data/training_smiles.pickle', 'rb') as inf:
        df = load(inf)
        df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

    total_molecules = len(df)  # total number of molecules you have
    batch_size = args.num_molecules # Use num_molecules argument as batch_size

    current_slice = args.slice # Initialize current_slice with the provided slice argument

    print(f"Starting epoch {args.epoch} from index {args.start_index} with batch size {batch_size}, initial slice {current_slice}")

    for start_index_loop in range(args.start_index, total_molecules, batch_size):
        print(f"Processing batch starting at index: {start_index_loop}, slice: {current_slice}")
        run_script(start_index_loop, batch_size, current_slice, args.epoch)
        current_slice += 1

if __name__ == '__main__':
    main()
