import subprocess
import os
import sys
import argparse

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from tableone import TableOne
import seaborn as sns
import matplotlib.pyplot as plt
import os

import argparse
from itertools import combinations

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors

import numpy as np
from scipy.stats import entropy
from scipy.stats import gaussian_kde


import logging

plt.rcParams['font.family'] = 'Nimbus Sans'

# Define discrete features as per original script
DISCRETE = [
    'Ciclo3', 'Ciclo4', 'Ciclo5', 'Ciclo6', 'Ciclo7', 'Ciclo8', 'Ciclo9',
    'Ciclo10', 'Ciclo11', 'Ciclo12', 'Ciclo13', 'Ciclo14',
    'ComponentesConectados', 'Enlace_cuadrúples', 'NHOHCount', 'NOCount',
    'NumAliphaticRings', 'NumAromaticRings', 'NumHAcceptors',
    'NumHDonors', 'NumRotatableBonds', 'Otros ciclos', 'Plano',
    'RingCount', 'Valid'
]

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# List of 40 RDKit descriptors
DESCRIPTORS = ['BertzCT','MolLogP','MolWt','TPSA','NumHAcceptors','NumHDonors','NumRotatableBonds', 'NumAliphaticRings','NumAromaticRings']

# Function to calculate all descriptors
def calculate_descriptors(mol):
    descriptor_values = {}
    try:
        # Calculate all descriptors at once
        calc = Descriptors.CalcMolDescriptors(mol)
        
        # Add all descriptors from CalcMolDescriptors
        for desc_name in DESCRIPTORS:
            descriptor_values[desc_name] = calc[desc_name]


        
    except Exception as e:
        logger.error(f"Error calculating descriptors for molecule: {e}")
        # Assign NaN for all descriptors in case of error
        for desc in DESCRIPTORS:
            descriptor_values[desc] = np.nan

    
    return descriptor_values

# Function to get molecules from SMILES
def get_mols(smiles_list):
    molecs = []
    for i in smiles_list:
        mol = Chem.MolFromSmiles(i)
        if mol is not None:
            molecs.append(mol)
        else:
            molecs.append(None)  # Maintain index alignment
    return molecs

# Function to compute descriptors for a list of SMILES
from tqdm import tqdm

def compute_all_descriptors(smiles_list):
    mols = get_mols(smiles_list)
    descriptors = [calculate_descriptors(mol) if mol is not None else {desc: np.nan for desc in DESCRIPTORS} for mol in tqdm(mols, desc="Computing descriptors")]
    return pd.DataFrame(descriptors)

# Function to get fingerprints (if still needed)
def get_fingerprints(mols, radius=2, length=4096):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]

# Function to calculate internal pairwise similarities
def calculate_internal_pairwise_similarities(smiles_list):
    """
    Computes the pairwise similarities of the provided list of smiles against itself.

    Returns:
        Symmetric matrix of pairwise similarities. Diagonal is set to zero.
    """
    if len(smiles_list) > 10000:
        logger.warning(f'Calculating internal similarity on large set of SMILES strings ({len(smiles_list)})')
        logger.warning(f'Adapting the number of molecules to 10000')
        smiles_list = smiles_list[:10000]

    mols = get_mols(smiles_list)
    fps = get_fingerprints(mols)
    nfps = len(fps)

    similarities = np.zeros((nfps, nfps))

    for i in tqdm(range(1, nfps), desc="Calculating pairwise similarities"):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        similarities[i, :i] = sims
        similarities[:i, i] = sims

    print("similarities calculated")

    return similarities



# Argument parsing (unchanged)
parser = argparse.ArgumentParser(description='Compare molecule categories.')
parser.add_argument('-ori', action='store_true', help='Include ori in comparison')
parser.add_argument('-gen', action='store_true', help='Include gen in comparison')
parser.add_argument('-cero', action='store_true', help='Include cero in comparison')
parser.add_argument('-jtvae', action='store_true', help='Include jtvae in comparison')
parser.add_argument('-digress', action='store_true', help='Include digress in comparison')
parser.add_argument('-real', action='store_true', help='Include real in comparison')
parser.add_argument('-ref', '--reference', type=str, help='The reference category for comparison')
parser.add_argument('-dir', '--directory', nargs='+', help='Directory or list of directories to compare')
parser.add_argument('-load', '--load_descriptors', action='store_true', help='Load pre-calculated descriptors instead of recalculating them')
args = parser.parse_args()

# Check if at least one directory is provided
if not args.directory:
    raise ValueError("You must provide at least one directory for comparison.")

directories = args.directory

# Adjust the comparisons list based on the command line arguments
selected_categories = []
if args.ori:
    selected_categories.append('ori')
if args.gen:
    selected_categories.append('gen_' + directories[0])
if args.jtvae:
    selected_categories.append('jtvae')
if args.digress:
    selected_categories.append('digress')
if args.real:
    selected_categories.append('real')

contador = 1
if len(directories) > 1:
    while contador < len(directories):
        selected_categories.append('gen_' + directories[contador])
        contador += 1

# Check that the reference category is valid
if args.reference not in selected_categories:
    raise ValueError(f"The reference category '{args.reference}' is not among the selected categories.")

reference_category = args.reference

if len(selected_categories) not in [2, 3, 4, 5, 6, 7]:
    raise ValueError("You must select either two or three categories for comparison.")

# If more than two categories, generate all combinations
if len(selected_categories) != 2:
    comparisons = list(combinations(selected_categories, 2))
else:
    comparisons = [(selected_categories[0], selected_categories[1])]

str_date = directories[0]

if args.load_descriptors:
    # Load pre-calculated descriptors
    df_combined = pd.read_csv(f'mols_gen/{str_date}/combined_molecules_with_descriptors_100K.csv') # tambien version 100K
    
else:
    # Read CSV files
    df = pd.read_csv(f'mols_gen/{str_date}/all_generated_molecules.csv')
    df_jtvae = pd.read_csv(f'Data/jtvae_generated_filtered.csv')  # Updated path
    df_digress = pd.read_csv(f'Data/digress_generated_filtered.csv')  # Updated path

    df_molecules = pd.read_csv('Data/molecules_lt70atoms_annotated.csv')  # Updated path
    df_molecules = df_molecules.sample(frac=1, random_state=1111).reset_index(drop=True)

    # Select 100000 random molecules from the original dataset
    selected_smiles = df_molecules.sample(n=100000, random_state=1111)['smiles'].tolist()

    # Select 15000 random molecules from the generated dataset
    selected_smiles_gen = df.sample(n=15000, random_state=1111)['smiles'].tolist()

    # Separate the dataframe into original, generated, etc.
    df_ori = pd.DataFrame({'smiles_ori': selected_smiles})
    df_gen = pd.DataFrame({'smiles_gen': selected_smiles_gen})
    df_jtvae = df_jtvae[['smiles_jtvae']]
    df_digress = df_digress[['smiles_digress']]

    # Rename columns to have the same name across dataframes
    df_ori = df_ori.rename(columns={'smiles_ori': 'smiles'})
    df_gen = df_gen.rename(columns={'smiles_gen': 'smiles'})
    df_jtvae = df_jtvae.rename(columns={'smiles_jtvae': 'smiles'})
    df_digress = df_digress.rename(columns={'smiles_digress': 'smiles'})

    # Add a column to indicate the category
    df_ori['category'] = 'ori'
    df_gen['category'] = 'gen_' + str_date
    df_jtvae['category'] = 'jtvae'
    df_digress['category'] = 'digress'

    # Reset index for each DataFrame
    df_ori = df_ori.reset_index(drop=True)
    df_gen = df_gen.reset_index(drop=True)
    df_jtvae = df_jtvae.reset_index(drop=True)
    df_digress = df_digress.reset_index(drop=True)

    # Check if df_ori contains column time_pred
    if 'time_pred' in df_ori.columns:
        df_ori = df_ori.drop(columns=['time_pred'])
        df_gen = df_gen.drop(columns=['time_pred'])

    # Compute descriptors for each category DataFrame
    def add_descriptors(df_subset, smiles_column):
        
        descriptors_df = compute_all_descriptors(df_subset['smiles'])
        df_subset = pd.concat([df_subset, descriptors_df], axis=1)
        return df_subset

    df_ori = add_descriptors(df_ori, 'smiles')
    df_gen = add_descriptors(df_gen, 'smiles')
    df_jtvae = add_descriptors(df_jtvae, 'smiles')
    df_digress = add_descriptors(df_digress, 'smiles')

    # Concatenate all DataFrames
    df_combined = pd.concat([df_ori, df_gen, df_jtvae, df_digress], axis=0).reset_index(drop=True)

    # Save the combined dataframe with descriptors for future use
    df_combined.to_csv(f'mols_gen/{str_date}/combined_molecules_with_descriptors_100K.csv', index=False)


# If there are additional directories, process them
contador = 1
if len(directories) > 1:
    while contador < len(directories):
        df_new = pd.read_csv(f'mols_gen/{directories[contador]}/all_generated_molecules.csv')
        df_gen_new = df_new[['smiles_gen']]
        df_gen_new = df_gen_new.rename(columns={'smiles_gen': 'smiles'})
        df_gen_new['category'] = 'gen_' + directories[contador]
        df_gen_new = add_descriptors(df_gen_new, 'smiles')
        df_combined = pd.concat([df_combined, df_gen_new], axis=0).reset_index(drop=True)
        # Ensure output directory exists for additional directories
        os.makedirs(f'mols_gen/{directories[contador]}/graficas_guacamol_random_10K_composite', exist_ok=True)
        contador += 1

# Calculate internal similarity using the new descriptors or fingerprints if still needed
df_combined['internal_similarity'] = None
for categoria_similarity in selected_categories:
    # Get all smiles for this category
    category_smiles = df_combined[df_combined['category'] == categoria_similarity]['smiles']
    category_indices = df_combined[df_combined['category'] == categoria_similarity].index
    
    # If more than 10000 smiles, randomly sample 10000 and track their indices
    if len(category_smiles) > 10000:
        sample_indices = np.random.choice(len(category_smiles), 10000, replace=False)
        sampled_smiles = category_smiles.iloc[sample_indices]
        sims = calculate_internal_pairwise_similarities(sampled_smiles)
        
        # Calculate max similarities for sampled molecules
        sims_max = sims.max(axis=1)
        
        # Randomly assign these max similarities to all molecules in the category
        all_sims_max = np.random.choice(sims_max, size=len(category_smiles))
    else:
        sims = calculate_internal_pairwise_similarities(category_smiles)
        all_sims_max = sims.max(axis=1)
    
    df_combined.loc[category_indices, 'internal_similarity'] = all_sims_max

df_combined['internal_similarity'] = df_combined['internal_similarity'].astype(float)

# Save the combined dataframe (optional)
df_combined.to_csv(f'mols_gen/{str_date}/combined_molecules_with_descriptors.csv', index=False)

# Define the list of descriptors to compare
columns_to_compare = DESCRIPTORS.copy() + ['internal_similarity']

# Perform statistical comparisons using TableOne
for group1, group2 in comparisons:
    subset = df_combined[df_combined['category'].isin([group1, group2])]
    # Ensure all descriptors are present
    available_columns = [col for col in columns_to_compare if col in subset.columns]
    mytable = TableOne(
        subset,
        columns=available_columns,
        categorical=[],  # Define categorical columns if any among the new descriptors
        groupby='category',
        pval=True
    )
    print(f"\nComparison between {group1} and {group2}:")
    print(mytable.tabulate(tablefmt="grid"))
    mytable.to_csv(f'mols_gen/{str_date}/mytable_res_{group1}_vs_{group2}.csv')

# Identify categorical features based on the number of unique values
categorical_features = [
    feature for feature in columns_to_compare
    if df_combined[df_combined['category'].isin(selected_categories)][feature].nunique() <= 20
]

import numpy as np
from scipy.stats import entropy

def jensen_shannon_distance(distribution1, distribution2):
    """
    Calculate the Jensen-Shannon distance between two probability distributions.

    Parameters:
    - distribution1: Array-like, first probability distribution.
    - distribution2: Array-like, second probability distribution.

    Returns:
    - js_distance: Float, the Jensen-Shannon distance.
    """
    # Ensure the distributions sum to 1
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)
    
    # Calculate the average distribution
    avg_distribution = (distribution1 + distribution2) / 2
    
    # Calculate the Jensen-Shannon divergence
    js_divergence = (entropy(distribution1, avg_distribution) + entropy(distribution2, avg_distribution)) / 2
    
    # The square root of the divergence is the Jensen-Shannon distance
    js_distance = np.sqrt(js_divergence)

    return js_distance

from scipy.stats import gaussian_kde

def calculate_js_distances_with_kde_safe(df, categories, features, num_points=100, variance_threshold=1e-5):
    """
    Calculate the Jensen-Shannon distances for all features between each pair of categories using KDE,
    with added checks for low variance data.

    Parameters:
    - df: DataFrame, the combined dataset with all categories.
    - categories: List of strings, the categories to compare.
    - features: List of strings, the features for which to calculate the distances.
    - num_points: int, number of points to use in the KDE estimation.
    - variance_threshold: float, threshold below which the data is considered to have insufficient variance.

    Returns:
    - js_distances: DataFrame, containing the JS distances for each feature and category pair.
    """
    results = []

    for feature in features:
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                category1 = categories[i]
                category2 = categories[j]

                # Extract data
                data1 = df[df['category'] == category1][feature].dropna()
                data2 = df[df['category'] == category2][feature].dropna()

                # Check for sufficient variance and data points
                if data1.var() < variance_threshold or data2.var() < variance_threshold or len(data1) < 2 or len(data2) < 2:
                    results.append({
                    'Feature': feature,
                    'Category1': category1,
                    'Category2': category2,
                    'JS_Distance': 10
                })
                    continue  # Skip this feature for this pair of categories

                # Determine the common range for KDE
                min_range = min(data1.min(), data2.min())
                max_range = max(data1.max(), data2.max())
                kde_range = np.linspace(min_range, max_range, num_points)

                # Estimate density using KDE
                kde1 = gaussian_kde(data1)
                kde2 = gaussian_kde(data2)
                density1 = kde1(kde_range)
                density2 = kde2(kde_range)

                # Calculate JS distance
                js_distance = jensen_shannon_distance(density1, density2)

                results.append({
                    'Feature': feature,
                    'Category1': category1,
                    'Category2': category2,
                    'JS_Distance': js_distance
                })

    js_distances = pd.DataFrame(results)
    return js_distances


def calculate_js_distances_with_histograms(df, categories, features, cat_features, bins=30 ):
    """
    Calculate the Jensen-Shannon distances for all features between each pair of categories using histograms.

    Parameters:
    - df: DataFrame, the combined dataset with all categories.
    - categories: List of strings, the categories to compare.
    - features: List of strings, the features for which to calculate the distances.
    - bins: int or sequence, the bin specification for the histograms.

    Returns:
    - js_distances: DataFrame, containing the JS distances for each feature and category pair.
    """
    results = []

    for feature in features:
        
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                if feature in cat_features:
                    # Get unique values for the feature
                    unique_values = df[feature].dropna().unique()

                    # Initialize the probability distributions
                    prob_distributions = {category: [] for category in categories}

                    # Calculate the frequency of each unique value for each category
                    for category in categories:
                        category_data = df[df['category'] == category][feature]
                        total_count = len(category_data)
                        counts = category_data.value_counts()
                        for value in unique_values:
                            prob_distributions[category].append(counts.get(value, 0) / total_count)
                       
                    category1 = categories[i]
                    category2 = categories[j]

                    # Convert lists to numpy arrays
                    distribution1 = np.array(prob_distributions[category1])
                    distribution2 = np.array(prob_distributions[category2])

                    # Calculate JS distance
                    js_distance = jensen_shannon_distance(distribution1, distribution2)

                    results.append({
                        'Feature': feature,
                        'Category1': category1,
                        'Category2': category2,
                        'JS_Distance': js_distance
                    })
                else:
                    category1 = categories[i]
                    category2 = categories[j]

                    # Extract data for the two categories
                    data1 = df[df['category'] == category1][feature].dropna()
                    data2 = df[df['category'] == category2][feature].dropna()

                    # Calculate the histogram for each category
                    if not np.isfinite(data1).all():
                        print("Data contains non-finite values. Cleaning the data.")
                        # Remove non-finite values
                        data1 = data1[np.isfinite(data1)]
                    hist1, edges1 = np.histogram(data1, bins=bins, density=True)
                    hist2, edges2 = np.histogram(data2, bins=edges1, density=True)  # Use edges from the first histogram

                    # Calculate JS distance using the histogram bins
                    js_distance = jensen_shannon_distance(hist1, hist2)

                    results.append({
                        'Feature': feature,
                        'Category1': category1,
                        'Category2': category2,
                        'JS_Distance': js_distance
                    })

    js_distances = pd.DataFrame(results)
    return js_distances

# Calculate Jensen-Shannon distances using histograms with the new descriptors
js_distances_kde = calculate_js_distances_with_histograms(
    df_combined[df_combined['category'].isin(selected_categories)],
    selected_categories,
    columns_to_compare,
    categorical_features
)

# Ensure output directory exists
save_dir = f'mols_gen/{str_date}/graficas_guacamol_random_10K_composite'
os.makedirs(save_dir, exist_ok=True)

js_distances_kde.to_csv(f'{save_dir}/DistBetweenDists.csv')

# Plotting function remains unchanged but uses the new descriptors
def plot_feature_distributions_with_js(dataframe, features, js_distances, bins=30):
    import random
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    sns.set(style="white")  # Change from "whitegrid" to "white"
    plt.rcParams['font.family'] = 'Nimbus Sans'

    # Define descriptive names for features
    feature_names = {
        'BertzCT': 'Bertz Complexity Index',
        'MolLogP': 'Molecular LogP',
        'MolWt': 'Molecular Weight',
        'TPSA': 'Topological Polar Surface Area',
        'NumHAcceptors': 'Number of H-Bond Acceptors',
        'NumHDonors': 'Number of H-Bond Donors',
        'NumRotatableBonds': 'Number of Rotatable Bonds',
        'NumAliphaticRings': 'Number of Aliphatic Rings',
        'NumAromaticRings': 'Number of Aromatic Rings',
        'internal_similarity': 'Internal Similarity'
    }

    # Define styles for each category
    category_styles = {
        'ori': {
            'color': '#1b9e77',
            'linewidth': 2,
            'linestyle': '-',
            'fill': True,
            'alpha': 0.5
        },
        'gen_' + directories[0]: {
            'color': 'black',
            'linewidth': 4,
            'linestyle': '-', 
            'fill': False,
            'alpha': 1.0
        },
        'digress': {
            'color': '#d95f02',
            'linewidth': 2,
            'linestyle': '--',
            'fill': False,
            'alpha': 1.0
        },
        'jtvae': {
            'color': '#7570b3',
            'linewidth': 2,
            'linestyle': '--',
            'fill': False,
            'alpha': 1.0
        }
    }

    # Create the save directory
    save_dir = f"mols_gen/{directories[0]}/graficas_guacamol_random_10K_composite"
    os.makedirs(save_dir, exist_ok=True)

    # Use specific selected features instead of random selection
    selected_features = [
        'MolWt',              # Tier 1 - Strong win against both
        'MolLogP',            # Tier 1 - Clear win against both
        'internal_similarity', # Tier 1 - Good win against both
        'BertzCT',            # Tier 3 - Strong win against both
        'NumAromaticRings',   # Tier 3 - Win against JTVAE only
        'NumHDonors'          # Tier 2 - Lose against both
    ]

    # Create a figure with GridSpec: 2 rows and 4 columns. The last column will span both rows for the log JS ratios plot.
    fig = plt.figure(figsize=(30, 12)) # Increased figure size for better visualization
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1])

    # Create axes for the first 3 columns (6 subplots arranged in 2 rows x 3 columns)
    distribution_axes = []
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            distribution_axes.append(ax)

    # Create one axis for the log JS ratios plot in the last column spanning both rows
    ax_log = fig.add_subplot(gs[:, 3])

    # Plot distributions for each of the 6 randomly selected features
    for idx, (ax, feature) in enumerate(zip(distribution_axes, selected_features)):
        # Add letter label to each subplot
        letter = chr(97 + idx)  # a, b, c, etc.
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=28, 
                fontweight='bold', va='top', ha='left', fontname='Nimbus Sans')

        # Only show the legend in the first plot (idx==0)
        show_legend = (idx == 0)

        # For plots E and F (idx 4 and 5), position JS text higher
        js_y_base = 0.7 if idx >= 4 else 0.3

        if feature in DISCRETE:
            # For discrete features, use histogram
            all_data = dataframe[dataframe['category'].isin(selected_categories)][feature].dropna()
            if all_data.empty:
                bin_edges = np.arange(0, bins + 1)
            else:
                bin_min = all_data.min()
                bin_max = all_data.max()
                bin_edges = np.arange(np.floor(bin_min), np.ceil(bin_max) + 1)
            if len(bin_edges) == 1:
                bin_edges = np.append(bin_edges, bin_edges[0] + 1)
            
            # Plot categories in desired order
            plot_order = ['ori', 'gen_' + directories[0], 'digress', 'jtvae']
            for cat in plot_order:
                if cat not in selected_categories:
                    continue
                    
                data = dataframe[dataframe['category'] == cat][feature]
                if len(data) > 0:
                    style = category_styles.get(cat, {'color': 'gray', 'linewidth': 1, 'linestyle': '-', 'fill': True, 'alpha': 0.5})
                    js_values = js_distances[
                        (js_distances['Feature'] == feature) &
                        (((js_distances['Category1'] == cat) | (js_distances['Category2'] == cat)) &
                        ((js_distances['Category1'] == reference_category) | (js_distances['Category2'] == reference_category)))
                    ]['JS_Distance'].tolist()
                    js_formatted = ', '.join([f'{val:.3f}' for val in js_values])
                    if cat == reference_category:
                        label = "Original"
                    else:
                        if cat == 'gen_' + directories[0]:
                            label = f"CocoGraph"
                        elif cat == 'digress':
                            label = f"Digress"
                        elif cat == 'jtvae':
                            label = f"JTVAE"
                        else:
                            label = f"{cat}"
                    
                    # Add JS value as text in the graph for non-reference categories
                    if cat != reference_category and len(js_values) > 0:
                        js_text = f"JS: {js_values[0]:.3f}"
                        # Position in the right area, with slight vertical offsets to prevent overlap
                        # Using js_y_base to position higher for plots E and F
                        ax.text(0.95, js_y_base + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                    
                    sns.histplot(data=dataframe[dataframe['category'] == cat],
                                 x=feature,
                                 label=label,
                                 color=style['color'],
                                 linewidth=0 if cat == reference_category else style['linewidth'],
                                 linestyle=style['linestyle'],
                                 fill=style['fill'],
                                 alpha=style['alpha'],
                                 bins=bin_edges - 0.5,
                                 element="step" if not style['fill'] else "bars",
                                 kde=False,
                                 stat="density",
                                 common_norm=True,
                                 ax=ax)
                    ax.grid(False)  # Remove grid for each subplot
                    # Set tick colors to black
                    ax.tick_params(axis='both', colors='black')
                    for spine in ax.spines.values():
                        spine.set_color('black')
                    # Set descriptive x-axis label
                    ax.set_xlabel(feature_names.get(feature, feature))
                else:
                    print(f"No data for category {cat} in feature {feature}")
            if len(bin_edges) > 8:
                indices = np.linspace(0, len(bin_edges) - 1, 8, dtype=int)
                ticks = bin_edges[indices]
            else:
                ticks = bin_edges
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks.astype(int))
            ax.set_yscale('log')
            # Remove legend if this is not the first plot
            if ax.get_legend() and not show_legend:
                ax.get_legend().remove()
        else:
            # For continuous features, use KDE
            min_val = dataframe[dataframe['category'].isin(selected_categories)][feature].min()
            max_val = dataframe[dataframe['category'].isin(selected_categories)][feature].max()
            
            # Plot categories in desired order
            plot_order = [ 'gen_' + directories[0], 'digress', 'jtvae', 'ori']
            for cat in plot_order:
                if cat not in selected_categories:
                    continue
                    
                style = category_styles.get(cat, {'color': 'gray', 'linewidth': 1, 'linestyle': '-', 'fill': True, 'alpha': 0.5})
                if cat == reference_category:
                    label = "Original"
                else:
                    js_values = js_distances[
                        (js_distances['Feature'] == feature) &
                        (((js_distances['Category1'] == cat) | (js_distances['Category2'] == cat)) &
                         ((js_distances['Category1'] == reference_category) | (js_distances['Category2'] == reference_category)))
                    ]['JS_Distance'].tolist()
                    js_formatted = ', '.join([f'{val:.3f}' for val in js_values])
                    if cat == 'gen_' + directories[0]:
                        label = f"CoCoGraph"
                    elif cat == 'digress':
                        label = f"Digress"
                    elif cat == 'jtvae':
                        label = f"JTVAE"
                    else:
                        label = f"{cat}"
                
                # Add JS value as text in the graph for non-reference categories
                if cat != reference_category and len(js_values) > 0:
                    js_text = f"JS: {js_values[0]:.3f}"
                    # Position in the right area, with slight vertical offsets to prevent overlap
                    # Using js_y_base to position higher for plots E and F
                    ax.text(0.95, js_y_base + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                            js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                            color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                
                if feature == 'Enlaces_triples':
                    if cat == reference_category:
                        sns.kdeplot(data=np.log1p(dataframe[dataframe['category'] == cat][feature]),
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=0,
                                    alpha=style['alpha'],
                                    ax=ax)
                    else:
                        sns.kdeplot(data=np.log1p(dataframe[dataframe['category'] == cat][feature]),
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=style['linewidth'],
                                    alpha=style['alpha'],
                                    clip=(np.log1p(min_val), np.log1p(max_val)),
                                    ax=ax)
                else:
                    if cat == reference_category:
                        sns.kdeplot(data=dataframe[dataframe['category'] == cat][feature],
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=0,
                                    alpha=style['alpha'],
                                    ax=ax)
                    else:
                        sns.kdeplot(data=dataframe[dataframe['category'] == cat][feature],
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=style['linewidth'],
                                    alpha=style['alpha'],
                                    clip=(min_val, max_val),
                                    ax=ax)
                ax.grid(False)  # Remove grid for each subplot
                # Set tick colors to black
                ax.tick_params(axis='both', colors='black')
                for spine in ax.spines.values():
                    spine.set_color('black')
                # Set descriptive x-axis label
                ax.set_xlabel(feature_names.get(feature, feature))
            # Remove legend if this is not the first plot
            if ax.get_legend() and not show_legend:
                ax.get_legend().remove()

    # Add legend only to the first plot
    if len(distribution_axes) > 0:
        first_ax = distribution_axes[0]
        handles, labels = first_ax.get_legend_handles_labels()
        if handles:
            first_ax.legend(handles, labels, fontsize=22)

    # Now, create the log JS ratios plot on the last column axis (ax_log) with vertical orientation
    pd.set_option('display.max_columns', None)
    pd.reset_option('display.max_columns')
    js_ratios_digress = []
    js_ratios_jtvae = []
    for feat in features:
        js_our_model = js_distances[
            (js_distances['Feature'] == feat) &
            (js_distances['Category1'] == 'ori') &
            (js_distances['Category2'] == 'gen_' + directories[0])
        ]['JS_Distance'].values
        
        js_digress = js_distances[
            (js_distances['Feature'] == feat) &
            (js_distances['Category1'] == 'ori') &
            (js_distances['Category2'] == 'digress')
        ]['JS_Distance'].values
        
        js_jtvae = js_distances[
            (js_distances['Feature'] == feat) &
            (js_distances['Category1'] == 'ori') &
            (js_distances['Category2'] == 'jtvae')
        ]['JS_Distance'].values
        
        if len(js_our_model) > 0:
            js_our_model = np.where(js_our_model == 0, 0.0001, js_our_model)
            if len(js_digress) > 0:
                js_digress = np.where(js_digress == 0, 0.0001, js_digress)
                js_ratio = np.log2(js_digress / js_our_model)
                js_ratios_digress.extend(js_ratio)
            if len(js_jtvae) > 0:
                js_jtvae = np.where(js_jtvae == 0, 0.0001, js_jtvae)
                js_ratio = np.log2(js_jtvae / js_our_model)
                js_ratios_jtvae.extend(js_ratio)
    
    better_features_digress = int(np.sum(np.array(js_ratios_digress) > 0))
    worse_features_digress = int(np.sum(np.array(js_ratios_digress) < 0))
    equal_features_digress = int(np.sum(np.array(js_ratios_digress) == 0))
    better_features_jtvae = int(np.sum(np.array(js_ratios_jtvae) > 0))
    worse_features_jtvae = int(np.sum(np.array(js_ratios_jtvae) < 0))
    equal_features_jtvae = int(np.sum(np.array(js_ratios_jtvae) == 0))
    
    bins_vals = np.arange(-4, 4.25, 0.25)
    
    ax_log.axvspan(-4, 0, color='#e6e6e6', alpha=0.7)
    
    sns.histplot(js_ratios_jtvae, bins=bins_vals,
                 color=category_styles['jtvae']['color'],
                 alpha=0.5,
                 fill=True,
                 edgecolor=None,
                 label="JTVAE",
                 orientation="horizontal", ax=ax_log)
    ax_log.grid(False)  # Remove grid from log ratio plot
    sns.histplot(js_ratios_digress, bins=bins_vals,
                 color=category_styles['digress']['color'],
                 alpha=0.5,
                 fill=True,
                 edgecolor=None,
                 label="Digress",
                 orientation="horizontal", ax=ax_log)
    
    ax_log.set_xlabel('Performance log2 ratio, R')
    ax_log.set_ylabel('Frequency')
    
    # Add background text annotations with updated colors, smaller font size and increased opacity
    y_limit = ax_log.get_ylim()[1]
    ax_log.text(2, y_limit * 0.9, 'Better:', color='black', fontsize=24, ha='center', weight='bold', fontname='Nimbus Sans', zorder=0, alpha=1.0)
    ax_log.text(2, y_limit * 0.86, f"{better_features_jtvae} vs JTVAE", color=category_styles['jtvae']['color'], fontsize=24, ha='center', weight='bold', fontname='Nimbus Sans', zorder=0, alpha=1.0)
    ax_log.text(2, y_limit * 0.82, f"{better_features_digress} vs Digress", color=category_styles['digress']['color'], fontsize=24, ha='center', weight='bold', fontname='Nimbus Sans', zorder=0, alpha=1.0)
    ax_log.text(-2, y_limit * 0.9, 'Worse:', color='black', fontsize=24, ha='center', weight='bold', fontname='Nimbus Sans', zorder=0, alpha=1.0)
    ax_log.text(-2, y_limit * 0.86, f"{worse_features_jtvae} vs JTVAE", color=category_styles['jtvae']['color'], fontsize=24, ha='center', weight='bold', fontname='Nimbus Sans', zorder=0, alpha=1.0)
    ax_log.text(-2, y_limit * 0.82, f"{worse_features_digress} vs Digress", color=category_styles['digress']['color'], fontsize=24, ha='center', weight='bold', fontname='Nimbus Sans', zorder=0, alpha=1.0)

    sns.despine()
    
    # Add letter label to the log2 ratio plot (will be the last letter)
    last_letter = chr(97 + len(distribution_axes))  # After all distribution plots
    ax_log.text(0.02, 0.98, last_letter, transform=ax_log.transAxes, fontsize=28, 
               fontweight='bold', va='top', ha='left', fontname='Nimbus Sans')
    
    # Increase font sizes for legends, axis titles, and tick labels for all subplots
    for ax in fig.get_axes():
        # Increase title and axis label font sizes
        ax.title.set_fontsize(26)
        ax.xaxis.label.set_fontsize(26)
        ax.yaxis.label.set_fontsize(26)

        # Increase tick label font sizes
        for tick in ax.get_xticklabels():
            tick.set_fontsize(20)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(20)

        # Increase legend text font size if legend exists
        leg = ax.get_legend()
        if leg is not None:
            # Increase legend text size even more
            for text in leg.get_texts():
                text.set_fontsize(22)
            
            # Make legend title larger if it exists
            if leg.get_title():
                leg.get_title().set_fontsize(26)

    # Apply tighter layout with minimal padding for minimal white space
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
    
    # Save as both high-resolution PDF (for publication) and PNG (for quick viewing)
    pdf_path = os.path.join(save_dir, "grid_combined_plot.pdf")
    png_path = os.path.join(save_dir, "grid_combined_plot.png")
    
    # Save as PDF with minimal borders and ultra-high quality
    fig.savefig(pdf_path, format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    print(f"High-resolution PDF saved to: {pdf_path}")
    
    # Also save as PNG with higher resolution
    fig.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.05)
    print(f"High-resolution PNG saved to: {png_path}")
    
    plt.close(fig)


# Usage
plot_feature_distributions_with_js(df_combined, columns_to_compare, js_distances_kde)

# Define importance tiers for each feature
feature_tiers = {
    'MolWt': 1,        # Tier 1 - Most Important
    'MolLogP': 1,      # Tier 1
    'TPSA': 1,         # Tier 1
    'internal_similarity': 1, # Tier 1
    'NumHAcceptors': 2,  # Tier 2 - Moderately Important
    'NumHDonors': 2,     # Tier 2
    'NumRotatableBonds': 2, # Tier 2
    'BertzCT': 3,       # Tier 3 - Less Critical
    'NumAliphaticRings': 3, # Tier 3
    'NumAromaticRings': 3   # Tier 3
}

# Create DataFrame to store feature importance and log2 ratios
feature_importance_df = []

# Calculate log2 ratios for each feature and store in DataFrame
for feat in columns_to_compare:
    # Get JS distance for our model vs original
    js_our_model = js_distances_kde[
        (js_distances_kde['Feature'] == feat) &
        (js_distances_kde['Category1'] == 'ori') &
        (js_distances_kde['Category2'] == 'gen_' + str_date)
    ]['JS_Distance'].values
    
    # Get JS distance for digress vs original
    js_digress = js_distances_kde[
        (js_distances_kde['Feature'] == feat) &
        (js_distances_kde['Category1'] == 'ori') &
        (js_distances_kde['Category2'] == 'digress')
    ]['JS_Distance'].values
    
    # Get JS distance for jtvae vs original
    js_jtvae = js_distances_kde[
        (js_distances_kde['Feature'] == feat) &
        (js_distances_kde['Category1'] == 'ori') &
        (js_distances_kde['Category2'] == 'jtvae')
    ]['JS_Distance'].values
    
    # Calculate log2 ratios
    log2_vs_jtvae = np.nan
    log2_vs_digress = np.nan
    
    if len(js_our_model) > 0 and js_our_model[0] != 0:
        if len(js_jtvae) > 0 and js_jtvae[0] != 0:
            log2_vs_jtvae = np.log2(js_jtvae[0] / js_our_model[0])
        
        if len(js_digress) > 0 and js_digress[0] != 0:
            log2_vs_digress = np.log2(js_digress[0] / js_our_model[0])
    
    # Determine tier for this feature
    tier = feature_tiers.get(feat, 4)  # Default to tier 4 if not in the dictionary
    
    # Add to results
    feature_importance_df.append({
        'Feature': feat,
        'Importance_Tier': tier,
        'log2_vs_JTVAE': log2_vs_jtvae,
        'log2_vs_Digress': log2_vs_digress
    })

# Convert to DataFrame and sort by importance tier
feature_importance_df = pd.DataFrame(feature_importance_df)
feature_importance_df = feature_importance_df.sort_values(by=['Importance_Tier', 'Feature'])

# Ensure consistency in save_dir definition
save_dir = f"mols_gen/{directories[0]}/graficas_guacamol_random_10K_composite"
os.makedirs(save_dir, exist_ok=True)

# Save to CSV
feature_importance_df.to_csv(f'{save_dir}/feature_importance_log2_ratios.csv', index=False)
print(f"Feature importance and log2 ratios saved to: {save_dir}/feature_importance_log2_ratios.csv")

# Save JS distances CSV in the same directory as the generated figures
js_distances_kde.to_csv(f'{save_dir}/js_distances_kde.csv', index=False)
print(f"All JS distances saved to: {save_dir}/js_distances_kde.csv")
