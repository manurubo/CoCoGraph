"""
Visualize and Rank Inpainted Molecules

This script analyzes inpainted molecules and ranks them by descriptor distance to the original.
It creates a grouped visualization showing the top 10 candidates with:
1. Grid of molecule structures (original + top 10 candidates)
2. Single radar chart comparing all candidates (original shown with stars)
3. Each candidate in a unique color for easy identification

Features:
- Removes duplicate molecules (using canonical SMILES)
- Calculates 9 drug-like descriptors: MolWt, LogP, TPSA, QED, SA Score, etc.
- Computes Euclidean distance in normalized descriptor space
- Ranks all candidates by descriptor distance to original
- Creates grouped comparison visualization (top 10 on one radar chart)
- Exports ranked CSV files for further analysis
- Optional: individual visualizations with --create_summary flag

Usage:
    # Main usage (creates grouped visualization)
    python sample_scripts/visualize_inpainted_molecules.py \
        --input_dir Inpaint_Test \
        --top_n 10
    
    # With individual visualizations
    python sample_scripts/visualize_inpainted_molecules.py \
        --input_dir Inpaint_Test \
        --top_n 20 \
        --create_summary
    
    # Hide original molecule from grouped visualization
    python sample_scripts/visualize_inpainted_molecules.py \
        --input_dir Inpaint_Test \
        --top_n 10 \
        --hide_original
    
    # Show colored borders for debugging layout
    python sample_scripts/visualize_inpainted_molecules.py \
        --input_dir Inpaint_Test \
        --top_n 10 \
        --borders
"""

import os
import sys
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer
import math

plt.rcParams['font.family'] = 'Nimbus Sans'

# Define the 10 molecular descriptors for drug-like comparison
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

# Descriptive names for the radar chart
DESCRIPTOR_NAMES = {
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


def calculate_descriptors(smiles, original_smiles=None):
    """
    Calculate molecular descriptors for a SMILES string.
    
    Args:
        smiles (str): SMILES string
        original_smiles (str, optional): Unused; kept for backward compatibility
    
    Returns:
        dict: Dictionary with descriptor values
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
            
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        for desc in DESCRIPTORS:
            if desc not in descriptor_values:
                descriptor_values[desc] = np.nan
    
    return descriptor_values


# Fixed ranges for each descriptor based on typical drug-like molecules
# These allow meaningful comparison across different molecules
DESCRIPTOR_RANGES = {
    'MolWt': (0, 600),              # Molecular weight: 0-600 Da (Lipinski: <500)
    'MolLogP': (-5, 10),            # LogP: -5 (very hydrophilic) to 10 (very hydrophobic)
    'TPSA': (0, 200),               # TPSA: 0-200 Ų (typical for drug-like molecules)
    'NumHDonors': (0, 10),          # H-bond donors: 0-10 (Lipinski: <5)
    'NumHAcceptors': (0, 15),       # H-bond acceptors: 0-15 (Lipinski: <10)
    'NumAromaticRings': (0, 6),     # Aromatic rings: 0-6
    'QED': (0, 1),                  # Drug-likeness: 0-1 (higher is better)
    'SA_Score': (1, 10),            # Synthetic Accessibility: 1 (easy) - 10 (hard)
    'BertzCT': (0, 1000)            # Complexity: 0-1000 is typical range
}


def normalize_descriptors(original_values, generated_values):
    """
    Normalize descriptor values to 0-1 range using fixed scales for each descriptor.
    This allows comparison across different molecules on a consistent scale.
    
    Args:
        original_values (dict): Original molecule descriptors
        generated_values (dict): Generated molecule descriptors
    
    Returns:
        tuple: (normalized_original, normalized_generated)
    """
    normalized_original = {}
    normalized_generated = {}
    
    for desc in DESCRIPTORS:
        orig_val = original_values.get(desc, 0)
        gen_val = generated_values.get(desc, 0)
        
        # Handle NaN values
        if np.isnan(orig_val):
            orig_val = 0
        if np.isnan(gen_val):
            gen_val = 0
        
        # Get the fixed range for this descriptor
        min_val, max_val = DESCRIPTOR_RANGES[desc]
        range_val = max_val - min_val
        
        # Normalize to 0-1 based on fixed range
        # Values outside the range will be clipped to 0-1
        normalized_original[desc] = max(0, min(1, (orig_val - min_val) / range_val))
        normalized_generated[desc] = max(0, min(1, (gen_val - min_val) / range_val))
    
    return normalized_original, normalized_generated


def calculate_descriptor_distance(original_values, generated_values):
    """
    Calculate Euclidean distance between normalized descriptor vectors.
    Lower distance = more similar molecules.
    
    Args:
        original_values (dict): Original molecule descriptors
        generated_values (dict): Generated molecule descriptors
    
    Returns:
        float: Euclidean distance between molecules
    """
    norm_orig, norm_gen = normalize_descriptors(original_values, generated_values)
    
    # Calculate Euclidean distance
    distance = 0.0
    for desc in DESCRIPTORS:
        diff = norm_orig[desc] - norm_gen[desc]
        distance += diff * diff
    
    return math.sqrt(distance)


def create_radar_chart(ax, original_values, generated_values, fragment_formula='', show_values=True):
    """
    Create a radar chart comparing original and generated molecule descriptors.
    
    Args:
        ax: Matplotlib axis
        original_values (dict): Original molecule descriptors
        generated_values (dict): Generated molecule descriptors
        fragment_formula (str): Formula of the attached fragment
        show_values (bool): Whether to show actual values as text
    """
    # Number of variables
    num_vars = len(DESCRIPTORS)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Build centered values (original at 0.5, generated at 0.5 ± Δ where Δ = (gen-orig)/range)
    orig_values_display = []
    gen_values_display = []
    for desc in DESCRIPTORS:
        min_val, max_val = DESCRIPTOR_RANGES[desc]
        range_val = max_val - min_val
        orig_val = original_values.get(desc, 0)
        gen_val = generated_values.get(desc, 0)
        if np.isnan(orig_val):
            orig_val = 0
        if np.isnan(gen_val):
            gen_val = 0
        diff = (gen_val - orig_val) / range_val
        centered_val = 0.5 + diff
        centered_val = max(0, min(1, centered_val))
        orig_values_display.append(0.5)
        gen_values_display.append(centered_val)
    orig_values_display += orig_values_display[:1]
    gen_values_display += gen_values_display[:1]
    
    # Plot data
    ax.plot(angles, orig_values_display, 'o-', linewidth=2.5, label='Original', color='#1b9e77', markersize=6)
    ax.fill(angles, orig_values_display, alpha=0.25, color='#1b9e77')
    
    ax.plot(angles, gen_values_display, 's-', linewidth=2.5, label=f'+ {fragment_formula}', color='#d95f02', markersize=6)
    ax.fill(angles, gen_values_display, alpha=0.25, color='#d95f02')
    
    # Add actual values as text annotations if requested (at least 2 decimals)
    if show_values:
        for i, desc in enumerate(DESCRIPTORS):
            angle = angles[i]
            orig_val = original_values.get(desc, 0)
            gen_val = generated_values.get(desc, 0)
            
            # Format all values with at least 2 decimals
            orig_str = f"{orig_val:.2f}"
            gen_str = f"{gen_val:.2f}"
            
            # Position text further outside the plot to avoid overlap with labels
            text_radius = 1.30  # Increased from 1.15
            x_pos = angle
            
            # Combine both values with color coding
            text = f"{orig_str} → {gen_str}"
            ax.text(x_pos, text_radius, text, ha='center', va='center', 
                   fontsize=7.5, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    # Set labels with better positioning (moved further out)
    labels = [DESCRIPTOR_NAMES[desc] for desc in DESCRIPTORS]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=20, fontweight='bold')
    ax.tick_params(pad=20)  # Add padding to move labels away from center
    
    # Set y-axis to ±25% relative range around original (0.25..0.75)
    ax.set_ylim(0.25, 0.75)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['-25%', 'Original', '+25%'], size=20, color='black')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Add legend with better styling
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=22, frameon=True, shadow=True)
    
    # Add title
    title_text = "Descriptor Comparison (Δ from Original, ±25%)"
    ax.set_title(title_text, size=22, pad=50, fontweight='bold')


def visualize_molecule_pair(original_smiles, generated_smiles, fragment_formula, 
                            original_desc, generated_desc, save_path=None, distance=None, rank=None):
    """
    Create a visualization showing original molecule, generated molecule, and radar chart.
    
    Args:
        original_smiles (str): Original molecule SMILES
        generated_smiles (str): Generated molecule SMILES
        fragment_formula (str): Formula of attached fragment
        original_desc (dict): Original molecule descriptors
        generated_desc (dict): Generated molecule descriptors
        save_path (str): Path to save the figure
        distance (float, optional): Descriptor distance to original
        rank (int, optional): Rank of this candidate
    """
    # Create figure with 3 subplots (increased size for value annotations)
    fig = plt.figure(figsize=(20, 7))
    
    # Subplot 1: Original molecule
    ax1 = plt.subplot(1, 3, 1)
    mol_orig = Chem.MolFromSmiles(original_smiles)
    if mol_orig is not None:
        img_orig = Draw.MolToImage(mol_orig, size=(400, 400))
        ax1.imshow(img_orig)
        ax1.axis('off')
        ax1.set_title('Original Molecule', fontsize=22, fontweight='bold')
        # Add molecular weight below
        mw = original_desc.get('MolWt', 0)
        ax1.text(0.5, -0.05, f'MW: {mw:.1f}', transform=ax1.transAxes,
                ha='center', fontsize=18)
    else:
        ax1.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
        ax1.axis('off')
    
    # Subplot 2: Generated molecule
    ax2 = plt.subplot(1, 3, 2)
    mol_gen = Chem.MolFromSmiles(generated_smiles)
    if mol_gen is not None:
        img_gen = Draw.MolToImage(mol_gen, size=(400, 400))
        ax2.imshow(img_gen)
        ax2.axis('off')
        title = f'Generated Molecule (+ {fragment_formula})'
        if rank is not None:
            title = f'Rank #{rank}\n' + title
        ax2.set_title(title, fontsize=22, fontweight='bold')
        # Add molecular weight and distance below
        mw = generated_desc.get('MolWt', 0)
        info_text = f'MW: {mw:.1f}'
        if distance is not None:
            info_text += f' | Dist: {distance:.4f}'
        ax2.text(0.5, -0.05, info_text, transform=ax2.transAxes,
                ha='center', fontsize=18)
    else:
        ax2.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')
        ax2.axis('off')
    
    # Subplot 3: Radar chart
    ax3 = plt.subplot(1, 3, 3, projection='polar')
    create_radar_chart(ax3, original_desc, generated_desc, fragment_formula)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close(fig)


def create_summary_grid(df, output_dir, max_per_page=6):
    """
    Create a grid summary of multiple molecule pairs.
    
    Args:
        df (pd.DataFrame): DataFrame with molecule data
        output_dir (str): Directory to save visualizations
        max_per_page (int): Maximum molecules per summary page
    """
    num_molecules = len(df)
    num_pages = (num_molecules + max_per_page - 1) // max_per_page
    
    for page in range(num_pages):
        start_idx = page * max_per_page
        end_idx = min(start_idx + max_per_page, num_molecules)
        page_df = df.iloc[start_idx:end_idx]
        
        # Create figure with grid
        n_rows = (len(page_df) + 1) // 2  # 2 molecules per row
        fig = plt.figure(figsize=(20, 6 * n_rows))
        
        for idx, (mol_idx, row) in enumerate(page_df.iterrows()):
            # Calculate subplot position (2 columns x n_rows)
            row_pos = idx // 2
            col_pos = idx % 2
            
            # Original molecule
            ax_orig = plt.subplot(n_rows, 6, row_pos * 6 + col_pos * 3 + 1)
            mol_orig = Chem.MolFromSmiles(row['original_smiles'])
            if mol_orig is not None:
                img_orig = Draw.MolToImage(mol_orig, size=(300, 300))
                ax_orig.imshow(img_orig)
            ax_orig.axis('off')
            ax_orig.set_title(f'#{mol_idx} Original', fontsize=10)
            
            # Generated molecule
            ax_gen = plt.subplot(n_rows, 6, row_pos * 6 + col_pos * 3 + 2)
            mol_gen = Chem.MolFromSmiles(row['smiles'])
            if mol_gen is not None:
                img_gen = Draw.MolToImage(mol_gen, size=(300, 300))
                ax_gen.imshow(img_gen)
            ax_gen.axis('off')
            ax_gen.set_title(f'+ {row.get("fragment_formula", "?")}', fontsize=10)
            
            # Radar chart (without value annotations for cleaner summary view)
            ax_radar = plt.subplot(n_rows, 6, row_pos * 6 + col_pos * 3 + 3, projection='polar')
            original_desc = {desc: row[f'orig_{desc}'] for desc in DESCRIPTORS}
            generated_desc = {desc: row[f'gen_{desc}'] for desc in DESCRIPTORS}
            create_radar_chart(ax_radar, original_desc, generated_desc, 
                             row.get('fragment_formula', ''), show_values=False)
            ax_radar.set_title('', fontsize=8)  # Remove individual titles for compact view
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'summary_page_{page+1}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved summary page {page+1} to {save_path}")
        plt.close(fig)


def create_grouped_radar_visualization(df, output_dir, show_original=True, show_borders=False, input_dir_name=None):
    """
    Create a single visualization with all top N candidates on one radar chart.
    Original molecule shown with stars, each candidate with different color.
    
    Args:
        df (pd.DataFrame): DataFrame with top candidates (including descriptors)
        output_dir (str): Directory to save the visualization
        show_original (bool): Whether to show the original molecule in the visualization
        show_borders (bool): Whether to show colored borders around molecules for debugging
    """

    import matplotlib.cm as cm
    import matplotlib.gridspec as gridspec
    
    
    num_candidates = len(df)
    num_shown = min(10, num_candidates)
    # Create figure with custom GridSpec layout
    fig = plt.figure(figsize=(20, 18))
    #  Create GridSpec: 6 rows, 7 columns
    # Left side (columns 0-2): molecules (3 columns, square aspect)
    # Column 3: small gap
    # Right side (columns 4-6): radar chart (3 columns)
    # width_ratios: molecules columns are equal, gap is smaller, radar columns are equal
    # Use negative spacing to make molecules overlap and hide white space
    gs = gridspec.GridSpec(6, 7, figure=fig, hspace=-0.15, wspace=0,
                          left=0.02, right=0.98, top=0.98, bottom=0.02,
                          width_ratios=[1, 1, 1, 0.01, 1.0, 1.0, 1.0])
    # Colors for candidates (define early to use in molecule labels and radar chart)
    try:
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, min(num_candidates, 10)))
    except:
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']
    # Original molecule (top center of left side) - optional
    if show_original:
        ax_orig = fig.add_subplot(gs[0, 1])
        original_smi = df.iloc[0]['original_smiles']
        mol_orig = Chem.MolFromSmiles(original_smi)
        if mol_orig is not None:
            img_orig = Draw.MolToImage(mol_orig, size=(500, 500))
            ax_orig.imshow(img_orig, aspect='equal', interpolation='bilinear')
        ax_orig.set_aspect('equal')
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        # Add red border for debugging (optional)
        if show_borders:
            for spine in ax_orig.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
        else:
            ax_orig.axis('off')
        # Title inside the subplot at the top
        ax_orig.text(0.5, 0.98, 'ORIGINAL MOLECULE', fontsize=22, fontweight='bold', 
                    ha='center', va='top', transform=ax_orig.transAxes, color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red', linewidth=2))
        
        # Add star marker in the top right corner (outside the label)
        ax_orig.plot(0.98, 0.75, marker='*', markersize=25, color='red', 
                    transform=ax_orig.transAxes, markeredgecolor='darkred', markeredgewidth=1.5, zorder=10, clip_on=False)
        
        # Add molecular formula at the bottom (like candidate molecules)
        if mol_orig is not None:
            from rdkit.Chem import rdMolDescriptors
            mol_formula = rdMolDescriptors.CalcMolFormula(mol_orig)
            ax_orig.text(0.5, 0.25, f'{mol_formula}', fontsize=16, fontweight='bold',
                       ha='center', va='top', transform=ax_orig.transAxes,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                alpha=0.85, edgecolor='orange', linewidth=1.5),
                       color='darkblue', clip_on=False)
        
        # Top 10 candidates in a grid on left side (starting from row 1)
        molecule_positions = [
            (1, 0), (1, 1), (1, 2),  # Row 1: 3 molecules
            (2, 0), (2, 1), (2, 2),  # Row 2: 3 molecules
            (3, 0), (3, 1), (3, 2),  # Row 3: 3 molecules
            (4, 0),                   # Row 4: 1 molecule (only 10 total)
        ]
    else:
        # Top 10 candidates in a grid on left side (starting from row 0)
        molecule_positions = [
            (0, 0), (0, 1), (0, 2),  # Row 0: 3 molecules
            (1, 0), (1, 1), (1, 2),  # Row 1: 3 molecules
            (2, 0), (2, 1), (2, 2),  # Row 2: 3 molecules
            (3, 0),                   # Row 3: 1 molecule (only 10 total)
        ]
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= 10:  # Limit to 10
            break
        
        row_pos, col_pos = molecule_positions[i]
        ax_mol = fig.add_subplot(gs[row_pos, col_pos])
        
        mol_gen = Chem.MolFromSmiles(row['smiles'])
        if mol_gen is not None:
            img_gen = Draw.MolToImage(mol_gen, size=(500, 500))
            ax_mol.imshow(img_gen, aspect='equal', interpolation='bilinear')
        ax_mol.set_aspect('equal')
        ax_mol.set_xticks([])
        ax_mol.set_yticks([])
        # Add blue border for debugging (optional)
        if show_borders:
            for spine in ax_mol.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('blue')
                spine.set_linewidth(3)
        else:
            ax_mol.axis('off')
        
        # Add colored indicator box at previous title position (top-center) with rank and distance
        color_box_text = f"Rank #{i+1}  |  Dist: {row['distance']:.3f}"
        ax_mol.text(0.5, 0.98, color_box_text, fontsize=20, fontweight='bold',
                   ha='center', va='top', transform=ax_mol.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], 
                            alpha=0.9, edgecolor='black', linewidth=2),
                   color='white')
        
        # Add fragment formula at the bottom
        fragment = row.get('fragment_formula', 'N/A')
        ax_mol.text(0.5, 0.25, f'+ {fragment}', fontsize=16, fontweight='bold',
                   ha='center', va='top', transform=ax_mol.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                            alpha=0.85, edgecolor='orange', linewidth=1.5),
                   color='darkblue', clip_on=False)
        
        # Add overall title over the candidate molecules on the left side
        if i == 1:  # place once over the center molecule of the first candidate row
            suffix = ''
            if input_dir_name == 'Inpaint_Random_Paracetamol':
                suffix = ' inpainting small'
            elif input_dir_name == 'Inpaint_Random_Paracetamol_5to15':
                suffix = ' inpainting medium'
            ax_mol.text(0.5, 1.10, f'Top {num_shown} candidates{suffix}',
                       transform=ax_mol.transAxes, ha='center', va='bottom',
                       fontsize=26, fontweight='bold', color='black')
    
    # Create radar chart on the right side - same height as molecules
    # Start at same row as original (or row 0 if no original)
    if show_original:
        radar_start_row = 0
        radar_end_row = 3
    else:
        radar_start_row = 0
        radar_end_row = 2
    
    # Radar chart uses columns 4-6 (column 3 is the small gap)
    ax_radar = fig.add_subplot(gs[radar_start_row:radar_end_row, 4:7], projection='polar')
    
    # Add green border for debugging (optional)
    if show_borders:
        radar_bbox = ax_radar.get_position()
        rect = plt.Rectangle((radar_bbox.x0, radar_bbox.y0), radar_bbox.width, radar_bbox.height,
                             fill=False, edgecolor='green', linewidth=3, transform=fig.transFigure, 
                             zorder=1000, clip_on=False)
        fig.patches.append(rect)
    
    # Number of variables
    num_vars = len(DESCRIPTORS)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Get original molecule descriptors (they're the same for all rows)
    original_desc = {desc: df.iloc[0][f'orig_{desc}'] for desc in DESCRIPTORS}
    
    # Calculate original values
    orig_values_display = []
    for desc in DESCRIPTORS:
        # Center at 0.5 for all descriptors
        orig_values_display.append(0.5)
    orig_values_display += orig_values_display[:1]
    
    # Plot original with stars (no label needed)
    ax_radar.plot(angles, orig_values_display, '*-', linewidth=3,
                  color='red', markersize=15, markeredgewidth=2, markeredgecolor='darkred')
    
    # Plot each candidate
    for i, (idx, row) in enumerate(df.iterrows()):
        if i >= 10:  # Limit to 10
            break
            
        # Get descriptors for this candidate
        generated_desc = {desc: row[f'gen_{desc}'] for desc in DESCRIPTORS}
        
        # Calculate display values
        gen_values_display = []
        for desc in DESCRIPTORS:
            orig_val = original_desc[desc]
            gen_val = generated_desc[desc]
            min_val, max_val = DESCRIPTOR_RANGES[desc]
            range_val = max_val - min_val
            diff = (gen_val - orig_val) / range_val
            centered_val = 0.5 + diff
            centered_val = max(0, min(1, centered_val))
            gen_values_display.append(centered_val)
        
        gen_values_display += gen_values_display[:1]
        
        # Plot with unique color (no label needed - colored boxes on molecules show the mapping)
        ax_radar.plot(angles, gen_values_display, 'o-', linewidth=1.5, 
                     color=colors[i], markersize=5, alpha=0.8)
    
    # Set tick positions but hide default labels
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([])
    
    # Manually place labels with rotation to follow the circle
    labels = [DESCRIPTOR_NAMES[desc] for desc in DESCRIPTORS]
    for i, (label_text, angle) in enumerate(zip(labels, angles[:-1])):
        # Convert angle from radians to degrees
        rotation = np.degrees(angle)
        
        # Position label outside the plot
        label_distance = 0.77  # Distance from center
        
        # Adjust rotation and alignment so text reads outward from center
        if 90 < rotation <= 180:
            # Left side: flip to avoid upside-down text
            rotation = rotation - 180 + 90
        elif 0<=rotation <90:
            rotation = rotation -90
        else:
            rotation = rotation + 90
        
        # Place the text with centered alignment
        ax_radar.text(angle, label_distance, label_text, 
                     rotation=rotation, rotation_mode='anchor',
                     ha='center', va='center', 
                     fontsize=20, fontweight='bold')
    
    # Set y-axis with original centered at 0.5 and show only ±25%
    ax_radar.set_ylim(0.25, 0.75)
    ax_radar.set_yticks([0.25, 0.5, 0.75])
    ax_radar.set_yticklabels(['-25%', 'Original', '+25%'], size=20, color='black', fontweight='bold')
    ax_radar.grid(True, linestyle='--', alpha=0.7)
    
    # Legend removed - colored boxes on molecules show which color corresponds to which rank
    
    # Restore radar chart title
    ax_radar.set_title('Descriptor Comparison (Δ from Original, ±25%)', size=26, pad=25, fontweight='bold')
    
    # Don't use tight_layout since we have explicit GridSpec margins
    
    # Save
    save_path = os.path.join(output_dir, 'top10_grouped_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved grouped comparison to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Visualize inpainted molecules with descriptors')
    parser.add_argument('--input_dir', type=str, default='Inpaint_Test',
                       help='Directory containing the generated molecules (relative to mols_gen/)')
    parser.add_argument('--batch_file', type=str, default='all_generated_molecules.csv',
                       help='Name of the file to visualize (default: all_generated_molecules.csv)')
    parser.add_argument('--output_dir', type=str, default='',
                       help='Directory to save visualizations (default: input_dir/visualizations)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top candidates to evaluate (ranked by descriptor distance)')
    parser.add_argument('--create_summary', action='store_true',
                       help='Also create individual visualizations for each candidate (default: only grouped view)')
    parser.add_argument('--hide_original', action='store_true',
                       help='Hide the original molecule from the grouped visualization (default: show original)')
    parser.add_argument('--borders', action='store_true',
                       help='Show colored borders around molecules for debugging (default: no borders)')
    
    args = parser.parse_args()
    
    # Set output directory
    out_dir = os.path.join('mols_gen', args.input_dir, 'visualizations')
    if not args.output_dir:
        args.output_dir = out_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the CSV file
    # current directory
    print(f"Current directory: {os.getcwd()}")
    csv_path = os.path.join('mols_gen', args.input_dir, args.batch_file)
    print(f"Reading molecules from {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    print(f"Reading molecules from {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Total molecules found: {len(df)}")
    
    # Remove duplicates by converting to canonical SMILES
    print("Removing duplicates...")
    initial_count = len(df)
    
    # Add canonical SMILES column
    canonical_smiles = []
    for smiles in df['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))
        else:
            canonical_smiles.append(smiles)
    
    df['canonical_smiles'] = canonical_smiles
    
    # Remove duplicates based on canonical SMILES
    df = df.drop_duplicates(subset='canonical_smiles', keep='first').reset_index(drop=True)
    
    duplicates_removed = initial_count - len(df)
    print(f"  Removed {duplicates_removed} duplicates ({len(df)} unique molecules remaining)")
    
    print(f"Processing {len(df)} unique molecules to calculate descriptors and distances...")
    
    # Calculate descriptors for all molecules
    print("Calculating descriptors and distances...")
    distances = []
    for idx, row in df.iterrows():
        original_smi = row['original_smiles']
        generated_smi = row['smiles']
        
        # Calculate descriptors
        original_desc = calculate_descriptors(original_smi, original_smiles=None)
        generated_desc = calculate_descriptors(generated_smi, original_smiles=None)
        
        # Calculate distance
        distance = calculate_descriptor_distance(original_desc, generated_desc)
        distances.append(distance)
        
        # Add descriptors to dataframe
        for desc in DESCRIPTORS:
            df.at[idx, f'orig_{desc}'] = original_desc[desc]
            df.at[idx, f'gen_{desc}'] = generated_desc[desc]
        
        df.at[idx, 'distance'] = distance
    
    # Sort by distance (lowest = most similar)
    df = df.sort_values('distance').reset_index(drop=True)
    
    # Select top N candidates
    top_n = min(args.top_n, len(df))
    df_top = df.head(top_n)
    
    print(f"\n{'='*60}")
    print(f"RANKED CANDIDATES (Top {top_n} most similar to original)")
    print(f"{'='*60}")
    for i, (idx, row) in enumerate(df_top.iterrows()):
        print(f"Rank {i+1}: Distance = {row['distance']:.4f}, "
              f"Fragment = {row.get('fragment_formula', 'N/A')}")
    print(f"{'='*60}\n")
    
    # Create grouped radar visualization (main visualization)
    print(f"\nCreating grouped visualization for top {min(top_n, 10)} candidates...")
    create_grouped_radar_visualization(df_top.head(10), args.output_dir, 
                                      show_original=not args.hide_original,
                                      show_borders=args.borders,
                                      input_dir_name=args.input_dir)
    
    # Create individual visualizations for top N candidates (optional, if requested)
    if args.create_summary:
        print(f"\nCreating individual visualizations for top {top_n} candidates...")
        for rank, (idx, row) in enumerate(df_top.iterrows()):
            original_desc = {desc: row[f'orig_{desc}'] for desc in DESCRIPTORS}
            generated_desc = {desc: row[f'gen_{desc}'] for desc in DESCRIPTORS}
            
            # Include rank and distance in filename
            save_path = os.path.join(args.output_dir, 
                                     f'rank_{rank+1:02d}_dist_{row["distance"]:.4f}.png')
            visualize_molecule_pair(
                row['original_smiles'],
                row['smiles'],
                row.get('fragment_formula', 'Unknown'),
                original_desc,
                generated_desc,
                save_path,
                distance=row['distance'],
                rank=rank+1
            )
            print(f"  Saved rank {rank+1}: {save_path}")
    
    # Save descriptor comparison table for top N
    print("\nSaving descriptor comparison table...")
    comparison_data = []
    for rank, (idx, row) in enumerate(df_top.iterrows()):
        mol_data = {
            'Rank': rank + 1,
            'Distance': row['distance'],
            'SMILES': row['smiles'],
            'Original_SMILES': row['original_smiles'],
            'Fragment': row.get('fragment_formula', 'N/A')
        }
        for desc in DESCRIPTORS:
            mol_data[f'{desc}_Original'] = row[f'orig_{desc}']
            mol_data[f'{desc}_Generated'] = row[f'gen_{desc}']
            mol_data[f'{desc}_Change'] = row[f'gen_{desc}'] - row[f'orig_{desc}']
        comparison_data.append(mol_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(args.output_dir, 'top_candidates.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved top {top_n} candidates to {comparison_path}")
    
    # Also save all ranked molecules
    all_ranked_path = os.path.join(args.output_dir, 'all_ranked_molecules.csv')
    df.to_csv(all_ranked_path, index=False)
    print(f"Saved all ranked molecules to {all_ranked_path}")
    
    # Print summary statistics for top N
    print("\n" + "="*60)
    print(f"SUMMARY STATISTICS (Top {top_n} Candidates)")
    print("="*60)
    print(f"Total molecules loaded: {initial_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Unique molecules evaluated: {len(df)}")
    print(f"Top N visualized: {top_n}")
    print(f"Best distance (most similar): {df_top['distance'].min():.4f}")
    print(f"Worst distance (in top N): {df_top['distance'].max():.4f}")
    print(f"Average distance (top N): {df_top['distance'].mean():.4f}")
    # Removed similarity summary
    
    print("\nAverage descriptor changes (top N):")
    for desc in DESCRIPTORS:
        avg_change = comparison_df[f'{desc}_Change'].mean()
        avg_orig = comparison_df[f'{desc}_Original'].mean()
        avg_gen = comparison_df[f'{desc}_Generated'].mean()
        print(f"  {DESCRIPTOR_NAMES[desc]:18s}: {avg_orig:8.2f} → {avg_gen:8.2f} (Δ {avg_change:+.2f})")
    print("="*60)
    
    print(f"\n✓ Main grouped visualization: {os.path.join(args.output_dir, 'top10_grouped_comparison.png')}")
    print(f"✓ Top candidates CSV: {comparison_path}")
    print(f"✓ All ranked molecules: {all_ranked_path}")
    if args.create_summary:
        print(f"✓ Individual visualizations: {args.output_dir}/rank_*.png")


if __name__ == "__main__":
    main()

