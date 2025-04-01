import subprocess
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib_functions.libraries import *
from lib_functions.config import *

def run_script(start_index, num_molecules, subproceso, epoca):
    # Command to run the Python script with specific arguments
    command = [
        'python', 'main_scripts/main_single_fast_v11_ffnet_fps_finetune.py', # antes era 7
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
    

    with open('Data/TotalSmilesTogether.pickle', 'rb') as inf:
        df = load(inf)
        df = df.sample(frac=1, random_state=1111).reset_index(drop=True)

    total_molecules = len(df)  # total number of molecules you have
    batch_size = 100000        # number of molecules to process per batch
    epoca = 1

    subproceso = 20 # definir a 0 inicialmente  
    for start_index in range(subproceso*batch_size, total_molecules, batch_size):
        print(start_index)
        run_script(start_index, batch_size, subproceso, epoca)
        subproceso+=1

if __name__ == '__main__':
    main()
