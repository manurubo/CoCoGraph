"""
Analyze Novel Molecules Dataset

This script calculates molecular descriptors for all molecules in the novel_molecules.csv dataset.
It uses parallel processing to efficiently handle millions of molecules.

Features:
- Calculates 9 drug-like descriptors: MolWt, LogP, TPSA, QED, SA Score, etc.
- Parallel processing using multiprocessing (CPU cores - 4)
- Progress tracking with estimated time remaining
- Saves results to novel_molecules_descriptors.csv
- Creates distribution visualizations for all descriptors

Usage:
    python sample_scripts/analyze_novel_molecules.py
    
    # With custom number of workers
    python sample_scripts/analyze_novel_molecules.py --n_workers 8
    
    # Process only first N molecules (for testing)
    python sample_scripts/analyze_novel_molecules.py --limit 10000
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import math

plt.rcParams['font.family'] = 'Nimbus Sans'

# Paracetamol SMILES (reference molecule for comparison)
PARACETAMOL_SMILES = 'CC(=O)Nc1ccc(O)cc1'

# Define the molecular descriptors (same as visualize_inpainted_molecules.py)
DESCRIPTORS = [
    'MolWt',
    'MolLogP',
    'TPSA',
    'NumHDonors',
    'NumHAcceptors',
    'NumAromaticRings',
    'QED',
    'SA_Score',
    'BertzCT'
]

# Descriptive names for the plots
DESCRIPTOR_NAMES = {
    'MolWt': 'Molecular Weight (Da)',
    'MolLogP': 'LogP (Lipophilicity)',
    'TPSA': 'TPSA (Ų)',
    'NumHDonors': 'H-Bond Donors',
    'NumHAcceptors': 'H-Bond Acceptors',
    'NumAromaticRings': 'Aromatic Rings',
    'QED': 'Drug-likeness (QED)',
    'SA_Score': 'Synthetic Accessibility',
    'BertzCT': 'Molecular Complexity'
}

# Short names for radar chart (same as visualize_inpainted_molecules.py)
DESCRIPTOR_NAMES_SHORT = {
    'MolWt': 'Mol Weight',
    'MolLogP': 'LogP',
    'TPSA': 'TPSA',
    'NumHDonors': 'H Donors',
    'NumHAcceptors': 'H Acceptors',
    'NumAromaticRings': 'Aromatic Rings',
    'QED': 'Drug-likeness',
    'SA_Score': 'Synth. Access.',
    'BertzCT': 'Complexity'
}

# Fixed ranges for each descriptor (same as visualize_inpainted_molecules.py)
DESCRIPTOR_RANGES = {
    'MolWt': (0, 600),              # Molecular weight: 0-600 Da
    'MolLogP': (-5, 10),            # LogP: -5 to 10
    'TPSA': (0, 200),               # TPSA: 0-200 Ų
    'NumHDonors': (0, 10),          # H-bond donors: 0-10
    'NumHAcceptors': (0, 15),       # H-bond acceptors: 0-15
    'NumAromaticRings': (0, 6),     # Aromatic rings: 0-6
    'QED': (0, 1),                  # Drug-likeness: 0-1
    'SA_Score': (1, 10),            # Synthetic Accessibility: 1-10
    'BertzCT': (0, 1000)            # Complexity: 0-1000
}


def calculate_descriptors_single(smiles):
    """
    Calculate molecular descriptors for a single SMILES string.
    
    Args:
        smiles (str): SMILES string
    
    Returns:
        dict: Dictionary with descriptor values (or NaN if invalid)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {desc: np.nan for desc in DESCRIPTORS}
    
    descriptor_values = {}
    try:
        # Standard RDKit descriptors
        descriptor_values['MolWt'] = Descriptors.MolWt(mol)
        descriptor_values['MolLogP'] = Descriptors.MolLogP(mol)
        descriptor_values['TPSA'] = Descriptors.TPSA(mol)
        descriptor_values['NumHDonors'] = Descriptors.NumHDonors(mol)
        descriptor_values['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
        descriptor_values['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
        descriptor_values['BertzCT'] = Descriptors.BertzCT(mol)
        
        # QED (drug-likeness)
        descriptor_values['QED'] = QED.qed(mol)
        
        # Synthetic Accessibility Score
        descriptor_values['SA_Score'] = sascorer.calculateScore(mol)
        
        # No similarity calculation
            
    except Exception as e:
        # If any error occurs, set all to NaN
        for desc in DESCRIPTORS:
            if desc not in descriptor_values:
                descriptor_values[desc] = np.nan
    
    return descriptor_values


def calculate_descriptors_batch(smiles_list):
    """
    Calculate descriptors for a batch of SMILES strings.
    Used for parallel processing.
    
    Args:
        smiles_list (list): List of SMILES strings
    
    Returns:
        list: List of descriptor dictionaries
    """
    results = []
    for smiles in smiles_list:
        desc = calculate_descriptors_single(smiles)
        results.append(desc)
    return results


def normalize_descriptors(paracetamol_values, molecule_values):
    """
    Normalize descriptor values to 0-1 range using fixed scales for each descriptor.
    This allows comparison across different molecules on a consistent scale.
    (Same as visualize_inpainted_molecules.py)
    
    Args:
        paracetamol_values (dict): Paracetamol descriptors
        molecule_values (dict): Novel molecule descriptors
    
    Returns:
        tuple: (normalized_paracetamol, normalized_molecule)
    """
    normalized_paracetamol = {}
    normalized_molecule = {}
    
    for desc in DESCRIPTORS:
        paracetamol_val = paracetamol_values.get(desc, 0)
        molecule_val = molecule_values.get(desc, 0)
        
        # Handle NaN values
        if np.isnan(paracetamol_val):
            paracetamol_val = 0
        if np.isnan(molecule_val):
            molecule_val = 0
        
        # Get the fixed range for this descriptor
        min_val, max_val = DESCRIPTOR_RANGES[desc]
        range_val = max_val - min_val
        
        # Normalize to 0-1 based on fixed range
        # Values outside the range will be clipped to 0-1
        normalized_paracetamol[desc] = max(0, min(1, (paracetamol_val - min_val) / range_val))
        normalized_molecule[desc] = max(0, min(1, (molecule_val - min_val) / range_val))
    
    return normalized_paracetamol, normalized_molecule


def calculate_descriptor_distance(paracetamol_values, molecule_values):
    """
    Calculate Euclidean distance between normalized descriptor vectors.
    Lower distance = more similar molecules.
    (Same as visualize_inpainted_molecules.py)
    
    Args:
        paracetamol_values (dict): Paracetamol descriptors
        molecule_values (dict): Novel molecule descriptors
    
    Returns:
        float: Euclidean distance between molecules
    """
    norm_paracetamol, norm_molecule = normalize_descriptors(paracetamol_values, molecule_values)
    
    # Calculate Euclidean distance
    distance = 0.0
    for desc in DESCRIPTORS:
        diff = norm_paracetamol[desc] - norm_molecule[desc]
        distance += diff * diff
    
    return math.sqrt(distance)


def create_grouped_radar_visualization(df_top, paracetamol_descriptors, output_dir, show_original=True):
    """
    Create a grouped visualization similar to visualize_inpainted_molecules.py
    Shows paracetamol and top N most similar molecules.
    
    Args:
        df_top (pd.DataFrame): DataFrame with top N molecules (already sorted by distance)
        paracetamol_descriptors (dict): Paracetamol descriptor values
        output_dir (str): Directory to save the visualization
    """
    print("\nCreating grouped radar visualization...")
    
    # Use all provided molecules (limit to 10 for display purposes)
    num_to_show = min(len(df_top), 10)
    df_repr = df_top.head(num_to_show).copy()
    
    # Create figure with custom GridSpec layout (same as visualize_inpainted_molecules.py)
    fig = plt.figure(figsize=(20, 18))
    
    # GridSpec: 6 rows, 7 columns
    # Left side (columns 0-2): molecules grid
    # Column 3: small gap
    # Right side (columns 4-6): radar chart
    gs = gridspec.GridSpec(6, 7, figure=fig, hspace=-0.15, wspace=0,
                          left=0.02, right=0.98, top=0.98, bottom=0.02,
                          width_ratios=[1, 1, 1, 0.01, 1.0, 1.0, 1.0])
    
    # Colors for molecules (use tab10 colormap like visualize_inpainted_molecules.py)
    colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, num_to_show))
    
    # Paracetamol molecule (top center of left side) - same as visualize_inpainted_molecules.py
    if show_original:
        ax_orig = fig.add_subplot(gs[0, 1])
        paracetamol_mol = Chem.MolFromSmiles(PARACETAMOL_SMILES)
        if paracetamol_mol is not None:
            img_orig = Draw.MolToImage(paracetamol_mol, size=(500, 500))
            ax_orig.imshow(img_orig, aspect='equal', interpolation='bilinear')
        ax_orig.set_aspect('equal')
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        ax_orig.axis('off')
        
        # Title for paracetamol
        ax_orig.text(0.5, 0.98, 'PARACETAMOL', fontsize=22, fontweight='bold', 
                    ha='center', va='top', transform=ax_orig.transAxes, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red', linewidth=2))
        
        # Add red star indicator (like original molecule in visualize_inpainted_molecules.py)
        ax_orig.text(0.98, 0.98, '★', fontsize=16, fontweight='bold',
                    ha='right', va='top', transform=ax_orig.transAxes,
                    bbox=dict(boxstyle='square,pad=0.4', facecolor='red', 
                             alpha=0.9, edgecolor='black', linewidth=2),
                    color='white')
        
        # Add paracetamol molecular formula at the bottom
        paracetamol_mol = Chem.MolFromSmiles(PARACETAMOL_SMILES)
        if paracetamol_mol:
            paracetamol_formula = rdMolDescriptors.CalcMolFormula(paracetamol_mol)
            ax_orig.text(0.5, 0.80, paracetamol_formula, fontsize=10, fontweight='bold',
                        ha='center', va='bottom', transform=ax_orig.transAxes,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                 alpha=0.85, edgecolor='orange', linewidth=1.5),
                        color='darkblue')
    
    # Top N most similar molecules in a grid on left side (starting from row 1)
    # Generate positions dynamically based on num_to_show
    molecule_positions = []
    start_row = 1 if show_original else 0
    for i in range(num_to_show):
        row_pos = start_row + (i // 3)  # 3 molecules per row
        col_pos = i % 3
        molecule_positions.append((row_pos, col_pos))
    
    for i, (idx, row) in enumerate(df_repr.iterrows()):
        if i >= num_to_show:
            break
        
        row_pos, col_pos = molecule_positions[i]
        ax_mol = fig.add_subplot(gs[row_pos, col_pos])
        
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is not None:
            img = Draw.MolToImage(mol, size=(500, 500))
            ax_mol.imshow(img, aspect='equal', interpolation='bilinear')
        ax_mol.set_aspect('equal')
        ax_mol.set_xticks([])
        ax_mol.set_yticks([])
        ax_mol.axis('off')
        
        # Add centered colored indicator box with rank and distance (match other script)
        color_box_text = f"Rank #{i+1}  |  Dist: {row['distance']:.3f}"
        ax_mol.text(0.5, 0.98, color_box_text, fontsize=20, fontweight='bold',
                   ha='center', va='top', transform=ax_mol.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                            alpha=0.9, edgecolor='black', linewidth=2),
                   color='white')
        
        # Add molecular formula at bottom (moved down)
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol:
            formula = rdMolDescriptors.CalcMolFormula(mol)
            ax_mol.text(0.5, 0.25, formula, fontsize=16, fontweight='bold',
                       ha='center', va='top', transform=ax_mol.transAxes,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.85, edgecolor='orange', linewidth=1.5),
                       color='darkblue', clip_on=False)

        # Add overall title over the candidate molecules on the left side (center of first row)
        if i == 1:
            ax_mol.text(0.5, 1.10, f'Top {num_to_show} candidates from database',
                       transform=ax_mol.transAxes, ha='center', va='bottom',
                       fontsize=26, fontweight='bold', color='black')
    
    # Create radar chart on the right side (match height based on show_original)
    if show_original:
        ax_radar = fig.add_subplot(gs[0:3, 4:7], projection='polar')
    else:
        ax_radar = fig.add_subplot(gs[0:2, 4:7], projection='polar')
    
    # Number of variables
    num_vars = len(DESCRIPTORS)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Calculate paracetamol display values: center at 0.5 for all descriptors
    paracetamol_values_display = []
    for desc in DESCRIPTORS:
        paracetamol_values_display.append(0.5)
    paracetamol_values_display += paracetamol_values_display[:1]
    
    # Plot paracetamol with red stars (like original molecule in visualize_inpainted_molecules.py)
    ax_radar.plot(angles, paracetamol_values_display, '*-', linewidth=3,
                  color='red', markersize=15, markeredgewidth=2, markeredgecolor='darkred')
    
    # Plot each of the top N molecules (like candidates in visualize_inpainted_molecules.py)
    for i, (idx, row) in enumerate(df_repr.iterrows()):
        if i >= num_to_show:
            break
        
        # Calculate display values relative to paracetamol
        molecule_values_display = []
        for desc in DESCRIPTORS:
            paracetamol_val = paracetamol_descriptors[desc]
            molecule_val = row[desc]
            # Center at 0.5 relative to paracetamol
            min_val, max_val = DESCRIPTOR_RANGES[desc]
            range_val = max_val - min_val
            diff = (molecule_val - paracetamol_val) / range_val
            centered_val = 0.5 + diff
            centered_val = max(0, min(1, centered_val))
            molecule_values_display.append(centered_val)
        
        molecule_values_display += molecule_values_display[:1]
        
        ax_radar.plot(angles, molecule_values_display, 'o-', linewidth=1.5, 
                     color=colors[i], markersize=5, alpha=0.8)
    
    # Set tick positions but hide default labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([])
    
    # Manually place labels with rotation (same as visualize_inpainted_molecules.py)
    labels = [DESCRIPTOR_NAMES_SHORT[desc] for desc in DESCRIPTORS]
    for i, (label_text, angle) in enumerate(zip(labels, angles[:-1])):
        rotation = np.degrees(angle)
        label_distance = 0.77
        
        # Adjust rotation for readability
        if 90 < rotation <= 180:
            rotation = rotation - 180 + 90
        elif 0 <= rotation < 90:
            rotation = rotation - 90
        else:
            rotation = rotation + 90
        
        ax_radar.text(angle, label_distance, label_text, 
                     rotation=rotation, rotation_mode='anchor',
                     ha='center', va='center', 
                     fontsize=20, fontweight='bold')
    
    # Set y-axis with paracetamol centered at 0.5 and show only ±25%
    ax_radar.set_ylim(0.25, 0.75)
    ax_radar.set_yticks([0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(['-25%', 'Original', '+25%'], size=20, color='black', fontweight='bold')
    ax_radar.grid(True, linestyle='--', alpha=0.7)
    
    # No legend needed - colored boxes on molecules show the mapping
    
    # Add title similar to visualize_inpainted_molecules.py
    ax_radar.set_title('Descriptor Comparison (Δ from Original, ±25%)', 
                      size=26, pad=25, fontweight='bold')
    
    # Save
    save_path = os.path.join(output_dir, 'grouped_radar_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved grouped radar visualization to {save_path}")
    plt.close(fig)


def create_distribution_plots(df, output_dir):
    """
    Create distribution plots for all molecular descriptors.
    
    Args:
        df (pd.DataFrame): DataFrame with descriptor columns
        output_dir (str): Directory to save plots
    """
    print("\nCreating distribution visualizations...")
    
    # Create a figure with subplots for all descriptors
    n_descriptors = len(DESCRIPTORS)
    n_cols = 3
    n_rows = (n_descriptors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()
    
    for i, desc in enumerate(DESCRIPTORS):
        ax = axes[i]
        
        # Get data (remove NaN values)
        data = df[desc].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            ax.set_title(DESCRIPTOR_NAMES[desc])
            continue
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        
        # Create histogram
        n_bins = min(100, int(np.sqrt(len(data))))
        counts, bins, patches = ax.hist(data, bins=n_bins, edgecolor='black', 
                                       alpha=0.7, color='steelblue')
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        
        # Set labels and title
        ax.set_xlabel(DESCRIPTOR_NAMES[desc], fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{DESCRIPTOR_NAMES[desc]}\n(μ={mean_val:.2f}, σ={std_val:.2f})', 
                    fontsize=12, fontweight='bold')
        
        # Format y-axis to show raw numbers
        ax.ticklabel_format(style='plain', axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Start y-axis at 0
        ax.set_ylim(bottom=0)
        
        # Add legend
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_descriptors, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, 'descriptor_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plots to {save_path}")
    plt.close(fig)
    
    # Create individual high-resolution plots for each descriptor
    print("\nCreating individual descriptor plots...")
    individual_dir = os.path.join(output_dir, 'individual_descriptors')
    os.makedirs(individual_dir, exist_ok=True)
    
    for desc in DESCRIPTORS:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data (remove NaN values)
        data = df[desc].dropna()
        
        if len(data) == 0:
            continue
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        std_val = data.std()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        
        # Create histogram
        n_bins = min(100, int(np.sqrt(len(data))))
        counts, bins, patches = ax.hist(data, bins=n_bins, edgecolor='black', 
                                       alpha=0.7, color='steelblue')
        
        # Add vertical lines for statistics
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2.5, 
                  label=f'Median: {median_val:.2f}')
        ax.axvline(q25, color='orange', linestyle=':', linewidth=2, 
                  label=f'Q25: {q25:.2f}')
        ax.axvline(q75, color='purple', linestyle=':', linewidth=2, 
                  label=f'Q75: {q75:.2f}')
        
        # Set labels and title
        ax.set_xlabel(DESCRIPTOR_NAMES[desc], fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Molecules', fontsize=13, fontweight='bold')
        ax.set_title(f'Distribution of {DESCRIPTOR_NAMES[desc]}\n'
                    f'Mean={mean_val:.2f}, Median={median_val:.2f}, StdDev={std_val:.2f}',
                    fontsize=14, fontweight='bold', pad=15)
        
        # Format axes to show raw numbers starting at 0
        ax.ticklabel_format(style='plain', axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        
        # Add legend
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(individual_dir, f'{desc}_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    print(f"✓ Saved individual plots to {individual_dir}/")


def create_summary_statistics(df, output_dir):
    """
    Create a summary statistics table for all descriptors.
    
    Args:
        df (pd.DataFrame): DataFrame with descriptor columns
        output_dir (str): Directory to save the summary
    """
    print("\nCalculating summary statistics...")
    
    stats_data = []
    for desc in DESCRIPTORS:
        data = df[desc].dropna()
        
        if len(data) > 0:
            stats = {
                'Descriptor': DESCRIPTOR_NAMES[desc],
                'Count': len(data),
                'Valid_Percent': len(data) / len(df) * 100,
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Q25': data.quantile(0.25),
                'Median': data.median(),
                'Q75': data.quantile(0.75),
                'Max': data.max()
            }
        else:
            stats = {
                'Descriptor': DESCRIPTOR_NAMES[desc],
                'Count': 0,
                'Valid_Percent': 0,
                'Mean': np.nan,
                'Std': np.nan,
                'Min': np.nan,
                'Q25': np.nan,
                'Median': np.nan,
                'Q75': np.nan,
                'Max': np.nan
            }
        
        stats_data.append(stats)
    
    stats_df = pd.DataFrame(stats_data)
    
    # Save to CSV
    stats_path = os.path.join(output_dir, 'descriptor_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved summary statistics to {stats_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("DESCRIPTOR STATISTICS")
    print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80)
    
    return stats_df


def main():
    parser = argparse.ArgumentParser(description='Analyze novel molecules dataset')
    parser.add_argument('--input_file', type=str, 
                       default='Data/generated_database/novel_molecules.csv',
                       help='Path to input CSV file with SMILES')
    parser.add_argument('--output_file', type=str,
                       default='Data/generated_database/novel_molecules_descriptors.csv',
                       help='Path to output CSV file with descriptors')
    parser.add_argument('--output_dir', type=str,
                       default='Data/generated_database/visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--n_workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 4)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of molecules to process (for testing)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top similar molecules to visualize (default: 10)')
    parser.add_argument('--hide_original', action='store_true',
                       help='Hide the original paracetamol panel in the grouped visualization')
    parser.add_argument('--batch_size', type=int, default=10000,
                       help='Batch size for parallel processing')
    parser.add_argument('--load_descriptors', action='store_true',
                       help='Load precomputed descriptors from output_file instead of recalculating')
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.n_workers is None:
        n_workers = max(1, cpu_count() - 4)
    else:
        n_workers = args.n_workers
    
    print(f"Using {n_workers} parallel workers (Total CPUs: {cpu_count()})")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    if args.load_descriptors and os.path.exists(args.output_file):
        print(f"\nLoading precomputed descriptors from {args.output_file}...")
        df = pd.read_csv(args.output_file)
        print(f"Loaded {len(df):,} rows with descriptors")
    else:
        # Read the CSV file
        print(f"\nReading molecules from {args.input_file}...")
        if not os.path.exists(args.input_file):
            print(f"Error: File not found: {args.input_file}")
            return
        
        df = pd.read_csv(args.input_file)
        print(f"Total molecules found: {len(df):,}")
    
    # Limit for testing
    if args.limit:
        df = df.head(args.limit)
        print(f"Limited to first {args.limit:,} molecules for testing")
    
    if not args.load_descriptors:
        # Get SMILES list
        smiles_list = df['smiles'].tolist()
        
        # Split into batches for parallel processing
        batch_size = args.batch_size
        n_batches = (len(smiles_list) + batch_size - 1) // batch_size
        batches = [smiles_list[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
        
        print(f"\nProcessing {len(smiles_list):,} molecules in {n_batches:,} batches...")
        print(f"Batch size: {batch_size:,}")
        
        # Process batches in parallel with progress bar
        start_time = time.time()
        all_results = []
        
        with Pool(n_workers) as pool:
            # Use imap for better progress tracking
            with tqdm(total=len(smiles_list), desc="Calculating descriptors") as pbar:
                for batch_results in pool.imap(calculate_descriptors_batch, batches):
                    all_results.extend(batch_results)
                    pbar.update(len(batch_results))
        
        elapsed_time = time.time() - start_time
        molecules_per_second = len(smiles_list) / elapsed_time
        
        print(f"\n✓ Processed {len(smiles_list):,} molecules in {elapsed_time:.1f} seconds")
        print(f"  ({molecules_per_second:.1f} molecules/second)")
        
        # Add descriptors to dataframe
        print("\nAdding descriptors to dataframe...")
        for desc in DESCRIPTORS:
            df[desc] = [result[desc] for result in all_results]
    
    # Count invalid molecules
    invalid_count = df[DESCRIPTORS].isna().all(axis=1).sum()
    print(f"Invalid molecules (all descriptors NaN): {invalid_count:,} ({invalid_count/len(df)*100:.2f}%)")
    
    # Calculate paracetamol descriptors (reference for comparison)
    print("\nCalculating paracetamol descriptors...")
    paracetamol_descriptors = calculate_descriptors_single(PARACETAMOL_SMILES)
    print(f"Paracetamol MW: {paracetamol_descriptors['MolWt']:.2f}, QED: {paracetamol_descriptors['QED']:.3f}")
    
    # Calculate distance to paracetamol for each molecule (like visualize_inpainted_molecules.py)
    print("\nCalculating similarity to paracetamol...")
    distances = []
    for idx, row in df.iterrows():
        molecule_desc = {desc: row[desc] for desc in DESCRIPTORS}
        distance = calculate_descriptor_distance(paracetamol_descriptors, molecule_desc)
        distances.append(distance)
    
    df['distance'] = distances
    
    # Save to CSV if we computed descriptors
    if not args.load_descriptors:
        print(f"\nSaving results to {args.output_file}...")
        df.to_csv(args.output_file, index=False)
        print(f"✓ Saved {len(df):,} molecules with descriptors (original order preserved)")
    
    # Create a sorted version for finding top similar molecules
    df_sorted = df.sort_values('distance').reset_index(drop=True)
    
    print(f"\nTop {args.top_n} most similar molecules to paracetamol:")
    for i in range(min(args.top_n, len(df_sorted))):
        print(f"  Rank {i+1}: Distance = {df_sorted.iloc[i]['distance']:.4f}, "
              f"SMILES = {df_sorted.iloc[i]['smiles']}")
    
    # Remove duplicates for visualization (using canonical SMILES)
    print(f"\nRemoving duplicates for visualization...")
    canonical_smiles_list = []
    for smiles in df_sorted['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles_list.append(Chem.MolToSmiles(mol, canonical=True))
        else:
            canonical_smiles_list.append(smiles)
    
    df_sorted['canonical_smiles'] = canonical_smiles_list
    initial_count = len(df_sorted)
    df_sorted_unique = df_sorted.drop_duplicates(subset='canonical_smiles', keep='first').reset_index(drop=True)
    duplicates_removed = initial_count - len(df_sorted_unique)
    print(f"  Removed {duplicates_removed} duplicates from visualization ({len(df_sorted_unique)} unique molecules)")
    
    # Create visualizations (using original order for distributions)
    create_distribution_plots(df, args.output_dir)
    
    # Create grouped radar visualization showing top N unique most similar to paracetamol
    df_top = df_sorted_unique.head(args.top_n)
    create_grouped_radar_visualization(df_top, paracetamol_descriptors, args.output_dir, show_original=not args.hide_original)
    
    # Create summary statistics
    create_summary_statistics(df, args.output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output file: {args.output_file}")
    print(f"Visualizations: {args.output_dir}/")
    #print(f"Total processing time: {elapsed_time:.1f} seconds")
    #print(f"Average speed: {molecules_per_second:.1f} molecules/second")
    print("="*80)


if __name__ == "__main__":
    main()

