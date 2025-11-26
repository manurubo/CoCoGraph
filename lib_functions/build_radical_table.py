import os
import sys
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import os

from lib_functions.valence_utils import build_radical_distribution_from_smiles_csv


def main() -> None:
    parser = argparse.ArgumentParser(description='Construye pesos por símbolo para radicales desde un origen de SMILES (.smiles uno por línea o CSV con columna smiles).')
    parser.add_argument('--input', required=False, default=os.path.join('Data', 'molecules_lt70atoms_annotated.csv'), help='Ruta a .smiles (uno por línea) o CSV con una columna smiles')
    parser.add_argument('--output', required=False, default=os.path.join('Data', 'radical_symbol_weights.json'), help='Ruta de salida JSON')
    parser.add_argument('--smiles_col', required=False, default='smiles', help='Nombre de la columna SMILES (ignorado para .smiles)')
    parser.add_argument('--chunksize', type=int, default=10000) 
    parser.add_argument('--max_molecules', type=int, default=1000000)
    parser.add_argument('--kekulize', action='store_true', help='Kekulizar antes de escanear radicales')
    args = parser.parse_args()

    dist = build_radical_distribution_from_smiles_csv(
        args.input,
        smiles_col=args.smiles_col,
        chunksize=args.chunksize,
        max_molecules=args.max_molecules,
        use_progress=True,
        kekulize=args.kekulize,
    )

    # Agregar por símbolo las ocurrencias con radical_electrons > 0 y normalizar a probabilidades
    symbol_counts = {}
    total_rad_atoms = 0
    for (sym, ch), counter in dist.items():
        # contar todos los estados rad>0
        rad_count = sum(cnt for rad, cnt in counter.items() if int(rad) > 0)
        if rad_count > 0:
            symbol_counts[sym] = symbol_counts.get(sym, 0) + rad_count
            total_rad_atoms += rad_count

    if total_rad_atoms <= 0:
        serializable = {}
    else:
        serializable = {sym: cnt / total_rad_atoms for sym, cnt in symbol_counts.items()}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(serializable, f, indent=2)


if __name__ == '__main__':
    main()


