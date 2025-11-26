import os
import sys
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import os

from lib_functions.valence_utils import build_charge_distribution_from_smiles_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Build charge distribution (+/- by element) from a SMILES source (.smiles one-per-line or CSV with a smiles column).')
    parser.add_argument('--input', required=False, default=os.path.join('Data', 'molecules_lt70atoms_annotated.csv'), help='Path to .smiles file (one SMILES per line) or CSV with a smiles column')
    parser.add_argument('--output', required=False, default=os.path.join('Data', 'charge_symbol_weights.json'), help='Output JSON path')
    parser.add_argument('--smiles_col', required=False, default='smiles', help='Name of the SMILES column (ignored for .smiles files)')
    parser.add_argument('--chunksize', type=int, default=10000)
    parser.add_argument('--max_molecules', type=int, default=1000000)
    parser.add_argument('--kekulize', action='store_true', help='Kekulize before scanning charges')
    args = parser.parse_args()

    dist = build_charge_distribution_from_smiles_csv(
        args.input,
        smiles_col=args.smiles_col,
        chunksize=args.chunksize,
        max_molecules=args.max_molecules,
        use_progress=True,
        kekulize=args.kekulize,
    )

    # Convert Counters to probability dicts per sign
    serializable = {}
    for sign, counter in dist.items():
        total = sum(counter.values())
        if total <= 0:
            serializable[sign] = {}
        else:
            serializable[sign] = {sym: cnt / total for sym, cnt in counter.items()}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2)


if __name__ == '__main__':
    main()




