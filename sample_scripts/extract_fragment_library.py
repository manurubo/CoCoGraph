"""
Extract Fragment Library from Molecules

This script analyzes a dataset of molecules and extracts common molecular fragments
that can be used for inpainting operations.

Fragments are extracted by:
1. Identifying common substructures
2. Breaking molecules at strategic bonds
3. Calculating molecular formulas of fragments

Usage:
    python sample_scripts/extract_fragment_library.py \
        --input_csv Data/molecules_lt70atoms_annotated.csv \
        --output_file Data/fragment_library.txt \
        --min_atoms 2 \
        --max_atoms 8 \
        --max_fragments 1000
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from collections import Counter
import argparse
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_all_fragments_from_molecule(mol, min_atoms=2, max_atoms=8):
    """
    Extract all possible fragments from a molecule by breaking single bonds.
    
    Args:
        mol: RDKit molecule
        min_atoms: Minimum number of heavy atoms in fragment
        max_atoms: Maximum number of heavy atoms in fragment
    
    Returns:
        list: List of molecular formulas for valid fragments
    """
    if mol is None:
        return []
    
    fragments = []
    
    # Get all single bonds (not in rings)
    single_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
            single_bonds.append(bond.GetIdx())
    
    # Try breaking each bond
    for bond_idx in single_bonds:
        try:
            # Break the bond
            fragmented = Chem.FragmentOnBonds(mol, [bond_idx], addDummies=False)
            
            # Get individual fragments
            frags = Chem.GetMolFrags(fragmented, asMols=True)
            
            for frag in frags:
                if frag is None:
                    continue
                
                # Count heavy atoms (non-hydrogen)
                num_heavy = frag.GetNumHeavyAtoms()
                
                if min_atoms <= num_heavy <= max_atoms:
                    # Get molecular formula
                    formula = rdMolDescriptors.CalcMolFormula(frag)
                    fragments.append(formula)
        
        except Exception:
            continue
    
    return fragments


def extract_common_substructures(smiles_list, min_atoms=2, max_atoms=6, sample_size=1000):
    """
    Extract common substructures from a list of molecules using RECAP fragmentation.
    
    Args:
        smiles_list: List of SMILES strings
        min_atoms: Minimum heavy atoms in fragment
        max_atoms: Maximum heavy atoms in fragment
        sample_size: Number of molecules to sample for analysis
    
    Returns:
        Counter: Dictionary of formula -> frequency
    """
    fragment_counter = Counter()
    
    # Sample molecules if dataset is large
    if len(smiles_list) > sample_size:
        smiles_sample = np.random.choice(smiles_list, sample_size, replace=False)
    else:
        smiles_sample = smiles_list
    
    print(f"Extracting fragments from {len(smiles_sample)} molecules...")
    
    for smi in tqdm(smiles_sample, desc="Fragmenting molecules"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Get fragments by breaking bonds
        fragments = get_all_fragments_from_molecule(mol, min_atoms, max_atoms)
        fragment_counter.update(fragments)
    
    return fragment_counter





def filter_and_rank_fragments(fragment_counter, min_frequency=5, max_fragments=1000):
    """
    Filter fragments by frequency and rank them.
    
    Args:
        fragment_counter: Counter object with fragment frequencies
        min_frequency: Minimum number of occurrences to keep
        max_fragments: Maximum number of fragments to return
    
    Returns:
        list: Sorted list of (formula, count) tuples
    """
    # Filter by minimum frequency
    filtered = {f: c for f, c in fragment_counter.items() if c >= min_frequency}
    
    # Sort by frequency (descending)
    sorted_fragments = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to max_fragments
    return sorted_fragments[:max_fragments]


def validate_fragment_formula(formula):
    """
    Check if a fragment formula is chemically reasonable.
    
    Args:
        formula: Molecular formula string
    
    Returns:
        bool: True if valid
    """
    try:
        # Try to create a molecule from the formula (basic check)
        # Just checking if RDKit can parse it
        from rdkit.Chem import rdMolDescriptors
        
        # Check if formula contains only valid elements
        valid_elements = {'C', 'H', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si'}
        
        # Simple parsing
        import re
        elements = re.findall(r'([A-Z][a-z]?)', formula)
        
        for elem in elements:
            if elem not in valid_elements:
                return False
        
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract molecular fragment library from dataset')
    parser.add_argument('--input_csv', type=str, default='Data/molecules_lt70atoms_annotated.csv',
                       help='Input CSV file with SMILES')
    parser.add_argument('--output_file', type=str, default='Data/fragment_library.txt',
                       help='Output file for fragment formulas')
    parser.add_argument('--min_atoms', type=int, default=2,
                       help='Minimum number of heavy atoms in fragment')
    parser.add_argument('--max_atoms', type=int, default=8,
                       help='Maximum number of heavy atoms in fragment')
    parser.add_argument('--min_frequency', type=int, default=5,
                       help='Minimum occurrence frequency for extracted fragments')
    parser.add_argument('--max_fragments', type=int, default=1000,
                       help='Maximum number of fragments to save')
    parser.add_argument('--sample_size', type=int, default=1000,
                       help='Number of molecules to sample for fragment extraction')

    
    args = parser.parse_args()
    
    print("="*60)
    print("Fragment Library Extraction")
    print("="*60)
    print(f"Input: {args.input_csv}")
    print(f"Output: {args.output_file}")
    print(f"Fragment size: {args.min_atoms}-{args.max_atoms} heavy atoms")
    print(f"Min frequency: {args.min_frequency}")
    print(f"Max fragments: {args.max_fragments}")
    print("="*60)
    
    # Read molecules
    print("\n1. Loading molecules...")
    df = pd.read_csv(args.input_csv)
    smiles_list = df['smiles'].tolist()
    print(f"   Loaded {len(smiles_list)} molecules")
    
    # Extract fragments from real molecules
    print("\n2. Extracting fragments from molecules...")
    fragment_counter = extract_common_substructures(
        smiles_list,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        sample_size=args.sample_size
    )
    print(f"   Found {len(fragment_counter)} unique fragments")
    
    # Filter and rank
    print("\n4. Filtering and ranking fragments...")
    ranked_fragments = filter_and_rank_fragments(
        fragment_counter,
        min_frequency=args.min_frequency,
        max_fragments=args.max_fragments
    )
    print(f"   Selected {len(ranked_fragments)} fragments")
    
    # Save to file
    print("\n5. Saving fragment library...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        f.write("# Molecular Fragment Library\n")
        f.write(f"# Generated from {args.input_csv}\n")
        f.write(f"# Total fragments: {len(ranked_fragments)}\n")
        f.write("#\n")
        f.write("# Format: FORMULA (frequency)\n")
        f.write("#\n")
        
        for formula, count in ranked_fragments:
            f.write(f"{formula}\n")
    
    print(f"   Saved to {args.output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total unique fragments: {len(fragment_counter)}")
    print(f"Fragments saved: {len(ranked_fragments)}")
    print("\nTop 20 most common fragments:")
    print("-" * 60)
    for i, (formula, count) in enumerate(ranked_fragments[:20], 1):
        print(f"{i:2d}. {formula:15s} (frequency: {count})")
    print("="*60)
    
    # Analyze fragment composition
    print("\nFragment composition:")
    sizes = []
    for formula, _ in ranked_fragments:
        mol = Chem.MolFromSmiles('')  # Dummy
        try:
            # Count heavy atoms from formula
            import re
            # Simple heavy atom count (not perfect but good enough)
            carbon_count = re.findall(r'C(\d*)', formula)
            c = int(carbon_count[0]) if carbon_count and carbon_count[0] else (1 if 'C' in formula else 0)
            nitrogen_count = re.findall(r'N(\d*)', formula)
            n = int(nitrogen_count[0]) if nitrogen_count and nitrogen_count[0] else (1 if 'N' in formula else 0)
            oxygen_count = re.findall(r'O(\d*)', formula)
            o = int(oxygen_count[0]) if oxygen_count and oxygen_count[0] else (1 if 'O' in formula else 0)
            
            heavy = c + n + o
            sizes.append(heavy)
        except:
            continue
    
    if sizes:
        print(f"  Size range: {min(sizes)}-{max(sizes)} heavy atoms")
        print(f"  Average size: {np.mean(sizes):.1f} heavy atoms")
        print(f"  Median size: {np.median(sizes):.1f} heavy atoms")
    
    print("\n✓ Fragment library generation complete!")
    print(f"\nUse with inpainting:")
    print(f"  python sample_scripts/sample_molecules_FPSmodel_inpaint.py \\")
    print(f"      --fragment_library {args.output_file} \\")
    print(f"      --target_smiles 'YOUR_SMILES'")


if __name__ == "__main__":
    main()

