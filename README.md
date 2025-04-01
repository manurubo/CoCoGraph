# Molecule Diffusion Project

This project has been reorganized into the following directory structure:

## Directory Structure

- `lib_functions/`: Contains core library functions and utilities
  - adjacency_utils.py
  - config.py
  - data_loader.py
  - data_preparation_utils.py
  - libraries.py
  - losses.py
  - models.py

- `main_scripts/`: Contains main sender scripts and training scripts
  - main_sender_*.py (sender scripts)
  - main_single_*.py and main_time_pred*.py (training scripts)

- `sample_scripts/`: Contains sample generation scripts
  - sample-fast-molecularformula-multimolecule*.py

- `compare_guacamol/`: Contains comparison and benchmarking scripts
  - compare*.py
  - Guacamol*.ipynb

## Usage

All import statements have been updated to reflect the new structure. When running scripts, make sure to run them from the project root directory, not from inside any of the subdirectories.

Example:
```bash
python3 main_scripts/main_sender_mainmodel.py
# or
cd /path/to/project/root
python3 sample_scripts/sample-fast-molecularformula-multimolecule_compartefm.py
```

## Additional Information

The imports in all Python files have been updated to correctly reference modules in their new locations. For example, imports from the `lib_functions` directory now use:

```python
from lib_functions.libraries import *
from lib_functions.config import *
```

If you encounter any import errors, please verify that you are running the scripts from the project root directory.

## Dependencies

You may need to install additional Python packages if they're not already available. Some modules may require:

```bash
pip install func-timeout
```

For a full list of dependencies, see the `requirements_vast.txt` file. 