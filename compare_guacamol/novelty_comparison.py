#!/usr/bin/env python3
"""
Script to compare generated molecules against a reference dataset (PubChem)
to determine novelty. Uses multiprocessing for efficient canonicalization.
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool, cpu_count
import sys
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
            mol = Chem.MolFromSmiles(smiles)
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
    print(f"Valid and UNIQUE SMILES: {len(valid_canonical)}")
    print(f"Non UniqueSMILES: {invalid_count}")
    
    return valid_canonical


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated molecules against reference dataset for novelty"
    )
    parser.add_argument(
        'generated_csv',
        type=str,
        help='Path to CSV file containing generated molecules (must have "smiles" column) | path to generated molecules file (one SMILES per line)' 
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='../Data/CID-SMILES-filtered-lt70-notraining.txt',
        help='Path to reference dataset file (default: ../Data/CID-SMILES-filtered-lt70-notraining.txt)'
    )
    parser.add_argument(
        '--num-processes',
        type=int,
        default=None,
        help='Number of processes to use (default: all available CPUs)'
    )

    parser.add_argument(
        '--load',
        action='store_true',
        help='Load reference from pre-canonicalized file (uses _canonical suffix)'
    )
    
    args = parser.parse_args()
    
    # Read reference dataset (PubChem)
    print(f"\nReading reference dataset from: {args.reference}")
    smiles_list_pc = []
    if args.load:
        canonical_file = args.reference.replace('.txt', '_canonical.txt')
        if not os.path.exists(canonical_file):
            print(f"Error: Canonical file not found: {canonical_file}")
            print("Please run without --load first to generate the canonical file.")
            sys.exit(1)
        with open(canonical_file, 'r') as f:
            for line in tqdm(f, desc="Reading canonical reference"):
                smiles_list_pc.append(line.strip())
    else:
        with open(args.reference, 'r') as f:
            for line in tqdm(f, desc="Reading reference"):
                smiles_list_pc.append(line.strip())
    
    print(f"Total reference molecules: {len(smiles_list_pc)}")
    if args.load:
        print(f"Already canonicalized")
    
    # Read generated molecules
    print(f"\nReading generated molecules from: {args.generated_csv}")
    try:
        if args.generated_csv.endswith('.csv'):
            df_gen = pd.read_csv(args.generated_csv)
            if 'smiles' not in df_gen.columns:
                print("Error: CSV file must contain a 'smiles' column")
                sys.exit(1)
            smiles_list_gen = df_gen['smiles'].tolist()
        else:
            if args.generated_csv.startswith('../Data/gdss') or args.generated_csv.startswith('../Data/GRUM'):
                with open(args.generated_csv, 'r') as f:
                    lines = f.readlines()
                    smiles_list_gen = []
                    for line in lines:
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            smiles = parts[0]
                            validity = int(parts[1])
                            if validity == 0:  # valid molecule
                                smiles_list_gen.append(smiles)
                            else:  # invalid molecule
                                smiles_list_gen.append("None")
            else:              
                with open(args.generated_csv, 'r') as f:
                    smiles_list_gen = [line.strip() for line in tqdm(f, desc="Reading generated molecules")]
    except FileNotFoundError:
        print(f"Error: Generated molecules file not found: {args.generated_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    print(f"Total generated molecules: {len(smiles_list_gen)}")

    # Filter SMILES with less than 70 atoms (including implicit H)
    filtered_smiles = []
    atom_counts = []
    for smiles in tqdm(smiles_list_gen):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Check if SMILES is valid
            # Add explicit hydrogens to count all atoms including implicit H
            mol_with_h = Chem.AddHs(mol)
            num_atoms = mol_with_h.GetNumAtoms()
            atom_counts.append(num_atoms)
            if num_atoms < 70:
                filtered_smiles.append(smiles)
        else:
            filtered_smiles.append(None)
    print(f"Total generated molecules with less than 70 atoms: {len(filtered_smiles)}")
    # Canonicalize reference dataset
    if not args.load:   
        print("\n" + "="*60)
        print("Canonicalizing reference dataset...")
        print("="*60)
        canonical_pc = canonicalize_smiles_list(
            smiles_list_pc,
            desc="Canonicalizing reference",
            num_processes=args.num_processes-4
        )
        canonical_file = args.reference.replace('.txt', '_canonical.txt')
        with open(canonical_file, 'w') as f:
            for smiles in canonical_pc:
                f.write(f"{smiles}\n")
        print(f"Saved canonical reference to: {canonical_file}")
    else:  
        canonical_pc = set(smiles_list_pc)
        print(f"Loaded canonical reference from: {canonical_file}")
    
    # Canonicalize generated molecules
    print("\n" + "="*60)
    print("Canonicalizing generated molecules...")
    print("="*60)
    canonical_gen = canonicalize_smiles_list(
        filtered_smiles,
        desc="Canonicalizing generated",
        num_processes=args.num_processes-4
    )
    
    # Calculate novelty
    print("\n" + "="*60)
    print("NOVELTY ANALYSIS")
    print("="*60)
    
    novel_molecules = canonical_gen - canonical_pc
    known_molecules = canonical_gen & canonical_pc
    
    novelty_rate = len(novel_molecules) / len(canonical_gen) * 100 if len(canonical_gen) > 0 else 0
    novelty_rate_vs_all = len(novel_molecules) / len(filtered_smiles) * 100 if len(filtered_smiles) > 0 else 0
    print(f"\nReference dataset (canonical): {len(canonical_pc)} molecules")
    print(f"Reference dataset (filtered): {len(filtered_smiles)} molecules")
    print(f"Generated molecules (canonical): {len(canonical_gen)} molecules")
    print(f"\nNovel molecules: {len(novel_molecules)}")
    print(f"Known molecules: {len(known_molecules)}")
    print(f"Novelty rate: {novelty_rate:.2f}%")
    print(f"Novelty rate vs all: {novelty_rate_vs_all:.2f}%")
    # Additional statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total generated (raw): {len(smiles_list_gen)}")
    print(f"Total generated (filtered): {len(filtered_smiles)}")
    print(f"Valid generated (canonical): {len(canonical_gen)}")
    print(f"Valid rate: {len(canonical_gen)/len(smiles_list_gen)*100:.2f}%")
    print(f"Novel molecules: {len(novel_molecules)}")
    print(f"Novelty rate: {novelty_rate:.2f}%")
    print(f"Novelty rate vs all: {novelty_rate_vs_all:.2f}%")
    


if __name__ == "__main__":
    main()

