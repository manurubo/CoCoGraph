import os
import sys
import math
import json
import argparse
import random
from datetime import datetime
from copy import deepcopy
from typing import Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import sqlite3

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib_functions.data_preparation_utils import smiles_to_graph
from lib_functions.adjacency_utils import connected_double_edge_swap, genera_intermedio, nx_to_rdkit


 


def simulate_noise_trajectory(graph, max_swaps: int, rng: random.Random):
    """Run connected_double_edge_swap to obtain the accumulated swap history.

    Returns: removed_edges_accumulated (list of accumulated swaps per step).
    """
    try:
        _, _, _, _, _, removed_edges_accumulated, _ = connected_double_edge_swap(
            deepcopy(graph), nswap=max_swaps, seed=rng
        )
        return removed_edges_accumulated
    except Exception:
        return []


def graph_after_k_swaps(base_graph, removed_edges_accumulated, k: int):
    if k <= 0 or not removed_edges_accumulated:
        return deepcopy(base_graph)
    k = min(k, len(removed_edges_accumulated))
    return genera_intermedio(deepcopy(base_graph), removed_edges_accumulated[k - 1])


def compute_edge_count(graph) -> int:
    # For MultiGraph, to_numpy_array without weight returns counts of parallel edges
    adj = None
    try:
        import networkx as nx
        adj = nx.to_numpy_array(graph, nodelist=list(graph.nodes()))
    except Exception:
        return max(1, graph.number_of_edges())
    return int(np.sum(adj) // 2)


def process_molecule(smiles: str, rng: random.Random, sigma_max: float) -> (pd.DataFrame, Dict[str, Any]):
    """Process one molecule following a noise schedule up to sigma_max.

    Returns the per-step descriptors and a summary dict with swaps info.
    """
    try:
        base_graph = smiles_to_graph(smiles)
        if base_graph is None:
            raise ValueError('Invalid graph from SMILES')
    except Exception:
        return pd.DataFrame(), {}

    total_edges = max(1, compute_edge_count(base_graph))
    sigma_max = float(max(0.0, sigma_max))
    target_swaps = int(math.ceil(sigma_max * total_edges))

    removed_edges_accumulated = simulate_noise_trajectory(base_graph, target_swaps, rng)
    total_steps = len(removed_edges_accumulated)

    # Always include step 0 (original)
    rows = []
    steps_to_eval = [0] + list(range(1, total_steps + 1))

    for step in steps_to_eval:
        if step == 0:
            g_k = deepcopy(base_graph)
        else:
            g_k = graph_after_k_swaps(base_graph, removed_edges_accumulated, step)

        try:
            mol_k = nx_to_rdkit(g_k, hidrogenos=False)
            noised_smiles = Chem.MolToSmiles(mol_k, isomericSmiles=True) if mol_k is not None else np.nan
        except Exception:
            noised_smiles = np.nan

        sigma_norm = 0.0 if total_steps == 0 else float(sigma_max) * (step / total_steps)
        row = {
            'smiles': smiles,
            'step': int(step),
            'total_steps': int(total_steps),
            'sigma_norm': float(sigma_norm),
            'edges': int(total_edges),
            'target_swaps': int(target_swaps),
            'noised_smiles': noised_smiles,
        }
        rows.append(row)

    # Swaps summary for this molecule
    summary = {
        'smiles': smiles,
        'edges': int(total_edges),
        'target_swaps_formula': f'ceil({sigma_max} * num_edges)',
        'target_swaps': int(target_swaps),
        'actual_swaps': int(total_steps)
    }

    return pd.DataFrame(rows), summary


# plotting/interactive functions removed


def main():
    parser = argparse.ArgumentParser(description='Noise molecules and emit original vs noised SMILES mapping (no property analysis).')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index for molecules')
    parser.add_argument('--num_molecules', type=int, default=1000, help='Number of molecules to process')
    parser.add_argument('--sigma_max', type=float, default=0.5, help='Maximum normalized sigma to simulate (e.g., 0.5 default, 1.0 doubles steps)')
    parser.add_argument('--input_csv', type=str, default='Data/molecules_lt70atoms_annotated.csv', help='Path to input CSV containing a smiles column')
    parser.add_argument('--smiles_column', type=str, default='smiles', help='Name of the column with SMILES in the CSV')
    parser.add_argument('--seed', type=int, default=1111, help='Random seed')
    parser.add_argument('--out_root', type=str, default=None, help='Output directory root; default timestamped under sample_scripts/output_YYYYmmdd_HHMMSS')
    args = parser.parse_args()

    # Output dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_root = os.path.join(os.path.dirname(__file__), 'output_' + timestamp)
    out_root = args.out_root if args.out_root is not None else default_root
    os.makedirs(out_root, exist_ok=True)

    # Read SMILES from CSV
    in_path = args.input_csv
    smiles_col = args.smiles_column
    df_in = pd.read_csv(in_path)
    if smiles_col not in df_in.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {in_path}. Available: {list(df_in.columns)}")
    smiles_series = df_in[smiles_col].astype(str).str.strip()
    smiles_df = pd.DataFrame({'smiles': smiles_series}).dropna().query("smiles != ''").drop_duplicates().sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    start = args.start_index
    end = min(start + args.num_molecules, smiles_df.shape[0])
    selected = smiles_df.iloc[start:end]['smiles'].tolist()

    all_rows = []
    swaps_summaries = []
    rng_master = random.Random(args.seed)

    for idx, sm in enumerate(tqdm(selected, desc='Processing molecules')):
        rng_i = random.Random(rng_master.randint(0, 2**31 - 1))
        df_mol, summary = process_molecule(sm, rng_i, sigma_max=args.sigma_max)
        if not df_mol.empty:
            df_mol.insert(0, 'mol_index', start + idx)
            all_rows.append(df_mol)
        if summary:
            print(f"Molecule {start + idx}: edges={summary['edges']} | target_swaps={summary['target_swaps']} (formula: {summary['target_swaps_formula']}) | actual_swaps={summary['actual_swaps']}")
            swaps_summaries.append({'mol_index': start + idx, **summary})

    if not all_rows:
        print('No valid molecules processed. Exiting.')
        return

    result_df = pd.concat(all_rows, axis=0, ignore_index=True)

    # Save swaps summary CSV
    swaps_csv = os.path.join(out_root, 'swaps_summary.csv')
    if swaps_summaries:
        pd.DataFrame(swaps_summaries).to_csv(swaps_csv, index=False)

    # Save noised SMILES mapping
    noised_cols = ['mol_index', 'smiles', 'step', 'total_steps', 'sigma_norm', 'noised_smiles']
    noised_df = result_df[noised_cols].copy()
    noised_csv = os.path.join(out_root, 'noised_smiles.csv')
    noised_df.to_csv(noised_csv, index=False)

    # Summary JSON
    summary = {
        'num_molecules': int(result_df['smiles'].nunique()),
        'num_records': int(result_df.shape[0]),
        'swaps_csv': swaps_csv,
        'noised_csv': noised_csv,
        'sigma_max': float(args.sigma_max),
        'input_csv': args.input_csv,
        'smiles_column': args.smiles_column,
    }
    with open(os.path.join(out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Saved noised SMILES to: {noised_csv}')
    print(f'Saved swaps summary to: {swaps_csv}')


if __name__ == '__main__':
    main()
