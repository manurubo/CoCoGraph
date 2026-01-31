#!/usr/bin/env python3
"""
Script to filter CID-SMILES file by removing all molecules that appear in training datasets.
This ensures that the benchmark reference dataset doesn't contain molecules used for training.

Input files:
  - CID-SMILES-filtered-lt70.txt: Main PubChem dataset to filter
  - molecules_lt70atoms_annotated.csv: Training molecules to remove
  - guacamol_v1_train.smiles: GuacaMol training set to remove
  - zinc250k.csv: ZINC250k training set to remove

Output:
  - CID-SMILES-filtered-lt70-notraining.txt: Filtered dataset
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import os


def canonicalize_smiles(smiles):
    """
    Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: SMILES string to canonicalize
        
    Returns:
        Canonical SMILES string or None if invalid
    """
    if isinstance(smiles, str):
        try:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
        except:
            pass
    return None


def canonicalize_smiles_list(smiles_list, desc="Canonicalizing", num_processes=None):
    """
    Canonicalize a list of SMILES using multiprocessing.
    
    Args:
        smiles_list: List of SMILES strings
        desc: Description for progress bar
        num_processes: Number of processes to use (default: cpu_count)
        
    Returns:
        Set of canonical SMILES (excluding invalid ones)
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes for canonicalization...")
    
    with Pool(num_processes) as pool:
        canonical_smiles = list(tqdm(
            pool.imap(canonicalize_smiles, smiles_list, chunksize=1000),
            total=len(smiles_list),
            desc=desc
        ))
    
    # Filter out None values (invalid SMILES)
    valid_canonical = set(s for s in canonical_smiles if s is not None)
    invalid_count = len(smiles_list) - len(valid_canonical)
    print(f"  Valid and UNIQUE SMILES: {len(valid_canonical)}")
    print(f"  Invalid/Duplicate SMILES: {invalid_count}")
    
    return valid_canonical


def canonicalize_smiles_list_keep_order(smiles_list, desc="Canonicalizing", num_processes=None):
    """
    Canonicalize a list of SMILES using multiprocessing, preserving order and duplicates.
    
    Args:
        smiles_list: List of SMILES strings
        desc: Description for progress bar
        num_processes: Number of processes to use (default: cpu_count)
        
    Returns:
        List of canonical SMILES (None for invalid ones)
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    print(f"Using {num_processes} processes for canonicalization...")
    
    with Pool(num_processes) as pool:
        canonical_smiles = list(tqdm(
            pool.imap(canonicalize_smiles, smiles_list, chunksize=1000),
            total=len(smiles_list),
            desc=desc
        ))
    
    valid_count = sum(1 for s in canonical_smiles if s is not None)
    invalid_count = len(canonical_smiles) - valid_count
    print(f"  Valid SMILES: {valid_count}")
    print(f"  Invalid SMILES: {invalid_count}")
    
    return canonical_smiles


def load_training_smiles(molecules_csv, guacamol_smiles, zinc_csv, num_processes):
    """
    Load and canonicalize all training SMILES from the three datasets.
    
    Returns:
        Set of canonical SMILES from all training sets
    """
    all_training_smiles = set()
    
    # Load molecules_lt70atoms_annotated.csv (SMILES in first column after header)
    print("\n" + "="*60)
    print(f"Loading {molecules_csv}...")
    print("="*60)
    df = pd.read_csv(molecules_csv)
    print(f"  Loaded {len(df)} molecules")
    canonical_molecules = canonicalize_smiles_list(
        df['smiles'].tolist(),
        desc="Canonicalizing molecules_lt70",
        num_processes=num_processes
    )
    all_training_smiles.update(canonical_molecules)
    print(f"  Added {len(canonical_molecules)} unique canonical SMILES")
    
    # Load guacamol_v1_train.smiles (one SMILES per line)
    print("\n" + "="*60)
    print(f"Loading {guacamol_smiles}...")
    print("="*60)
    with open(guacamol_smiles, 'r') as f:
        guacamol_list = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(guacamol_list)} molecules")
    canonical_guacamol = canonicalize_smiles_list(
        guacamol_list,
        desc="Canonicalizing GuacaMol",
        num_processes=num_processes
    )
    all_training_smiles.update(canonical_guacamol)
    print(f"  Added {len(canonical_guacamol)} unique canonical SMILES")
    
    # Load zinc250k.csv (SMILES in column 1, after header)
    print("\n" + "="*60)
    print(f"Loading {zinc_csv}...")
    print("="*60)
    df_zinc = pd.read_csv(zinc_csv)
    print(f"  Loaded {len(df_zinc)} molecules")
    canonical_zinc = canonicalize_smiles_list(
        df_zinc['smiles'].tolist(),
        desc="Canonicalizing ZINC250k",
        num_processes=num_processes
    )
    all_training_smiles.update(canonical_zinc)
    print(f"  Added {len(canonical_zinc)} unique canonical SMILES")
    
    return all_training_smiles


def filter_cid_smiles(cid_file, training_smiles, output_file, num_processes):
    """
    Filter CID-SMILES file to remove training molecules.
    
    Args:
        cid_file: Path to CID-SMILES-filtered-lt70.txt
        training_smiles: Set of canonical training SMILES to remove
        output_file: Path to save filtered SMILES
        num_processes: Number of processes for canonicalization
    """
    # Load CID-SMILES
    print("\n" + "="*60)
    print(f"Loading {cid_file}...")
    print("="*60)
    with open(cid_file, 'r') as f:
        cid_list = [line.strip() for line in f if line.strip()]
    print(f"  Loaded {len(cid_list)} molecules")
    
    # Canonicalize CID-SMILES (keep order to preserve original)
    print("\n" + "="*60)
    print("Canonicalizing CID-SMILES...")
    print("="*60)
    canonical_cid = canonicalize_smiles_list_keep_order(
        cid_list,
        desc="Canonicalizing CID-SMILES",
        num_processes=num_processes
    )
    
    # Filter out training molecules
    print("\n" + "="*60)
    print("Filtering out training molecules...")
    print("="*60)
    filtered_smiles = []
    removed_count = 0
    invalid_count = 0
    
    for original_smi, canonical_smi in tqdm(zip(cid_list, canonical_cid), 
                                             total=len(cid_list), 
                                             desc="Filtering"):
        if canonical_smi is None:
            invalid_count += 1
            continue
        
        if canonical_smi not in training_smiles:
            # Keep the original SMILES (not canonical) to preserve original format
            filtered_smiles.append(original_smi)
        else:
            removed_count += 1
    
    print(f"\nFiltering summary:")
    print(f"  Original CID-SMILES: {len(cid_list)}")
    print(f"  Invalid SMILES: {invalid_count}")
    print(f"  Removed (in training): {removed_count}")
    print(f"  Remaining: {len(filtered_smiles)}")
    print(f"  Removal rate: {removed_count/len(cid_list)*100:.2f}%")
    
    # Save filtered SMILES
    print("\n" + "="*60)
    print(f"Saving filtered SMILES to {output_file}...")
    print("="*60)
    with open(output_file, 'w') as f:
        for smiles in filtered_smiles:
            f.write(f"{smiles}\n")
    print(f"  Saved {len(filtered_smiles)} filtered SMILES")


def main():
    parser = argparse.ArgumentParser(
        description="Filter CID-SMILES file by removing training molecules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--cid-smiles',
        type=str,
        default='Data/CID-SMILES-filtered-lt70.txt',
        help='Path to CID-SMILES file to filter'
    )
    
    parser.add_argument(
        '--molecules-csv',
        type=str,
        default='Data/molecules_lt70atoms_annotated.csv',
        help='Path to molecules_lt70atoms_annotated.csv'
    )
    
    parser.add_argument(
        '--guacamol-smiles',
        type=str,
        default='Data/guacamol_v1_train.smiles',
        help='Path to guacamol_v1_train.smiles'
    )
    
    parser.add_argument(
        '--zinc-csv',
        type=str,
        default='Data/zinc250k.csv',
        help='Path to zinc250k.csv'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='Data/CID-SMILES-filtered-lt70-notraining.txt',
        help='Output file for filtered CID-SMILES'
    )
    
    parser.add_argument(
        '--num-processes',
        type=int,
        default=None,
        help='Number of processes to use (default: all CPUs)'
    )
    
    args = parser.parse_args()
    
    if args.num_processes is None:
        args.num_processes = cpu_count()
    
    print("="*60)
    print("CID-SMILES TRAINING DATA FILTER")
    print("="*60)
    print(f"CID-SMILES file: {args.cid_smiles}")
    print(f"Molecules CSV: {args.molecules_csv}")
    print(f"GuacaMol SMILES: {args.guacamol_smiles}")
    print(f"ZINC CSV: {args.zinc_csv}")
    print(f"Output file: {args.output}")
    print(f"Number of processes: {args.num_processes}")
    
    # Check input files exist
    for filepath in [args.cid_smiles, args.molecules_csv, args.guacamol_smiles, args.zinc_csv]:
        if not os.path.exists(filepath):
            print(f"ERROR: File not found: {filepath}")
            return
    
    # Load and canonicalize all training SMILES
    training_smiles = load_training_smiles(
        args.molecules_csv,
        args.guacamol_smiles,
        args.zinc_csv,
        args.num_processes
    )
    
    print("\n" + "="*60)
    print("TOTAL TRAINING MOLECULES")
    print("="*60)
    print(f"  Total unique training SMILES: {len(training_smiles)}")
    
    # Filter CID-SMILES
    filter_cid_smiles(
        args.cid_smiles,
        training_smiles,
        args.output,
        args.num_processes
    )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Filtered CID-SMILES saved to: {args.output}")
    print("\nYou can now use this file in your benchmarking notebook.")


if __name__ == '__main__':
    main()
