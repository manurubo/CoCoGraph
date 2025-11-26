import os
import sys
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import os
import json

from lib_functions.valence_utils import build_valence_distribution_from_smiles_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Build valid valence table from a SMILES source (.smiles one-per-line or CSV with a smiles column).')
    parser.add_argument('--input', required=False, default=os.path.join('Data', 'molecules_lt70atoms_annotated.csv'), help='.smiles file (one SMILES per line) or CSV with a smiles column')
    parser.add_argument('--output', required=False, default=os.path.join('Data', 'valid_valences.json'), help='Output JSON path')
    parser.add_argument('--smiles_col', required=False, default='smiles', help='Name of the SMILES column (ignored for .smiles files)')
    parser.add_argument('--include_h', action='store_true', help='Include implicit hydrogens in valence count')
    parser.add_argument('--aromatic_as_double', action='store_true', help='Count aromatic bonds as double (2)')
    parser.add_argument('--chunksize', type=int, default=10000)
    parser.add_argument('--max_molecules', type=int, default=100000)
    parser.add_argument('--no_kekulize', action='store_true', help='Disable kekulization before counting valences')
    args = parser.parse_args()

    dist = build_valence_distribution_from_smiles_csv(
        args.input,
        smiles_col=args.smiles_col,
        chunksize=args.chunksize,
        max_molecules=args.max_molecules,
        include_hydrogens=args.include_h,
        aromatic_as_double=args.aromatic_as_double,
        kekulize=not args.no_kekulize,
        count_implicit_hydrogen_atoms=True,
    )

    # Convert Counters to percentage dicts per (symbol,charge)
    serializable = {}
    for (sym, ch), counter in dist.items():
        total = sum(counter.values()) or 1
        serializable[f"{sym}__{ch}"] = {str(val): (count / total) for val, count in counter.items()}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2)


if __name__ == '__main__':
    main()


