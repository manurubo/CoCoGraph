import pandas as pd
from tableone import TableOne
import seaborn as sns
import matplotlib.pyplot as plt
import os

import argparse
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import Descriptors

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
DESCRIPTORS = [
    # 1. Basic Physicochemical Properties
    'MolWt', 'ExactMolWt', 'HeavyAtomCount', 'NumValenceElectrons',
    'NHOHCount', 'NOCount', 'FractionCSP3', 'qed', 'BalabanJ',
    # 2. Lipinski's Rule of Five Descriptors
    'NumHDonors', 'NumHAcceptors', 'MolLogP', 'NumRotatableBonds', 'TPSA',
    # 3. Ring and Aromaticity Descriptors
    'NumAromaticRings', 'NumAliphaticRings', 'RingCount', 
    'NumSaturatedRings',  'BertzCT',
    # 4. Electronic Descriptors
    'MolMR', 'MaxPartialCharge', 'MinPartialCharge',"MaxAbsPartialCharge",
    "MinAbsPartialCharge", 'Ipc', 'EState_VSA1',
    # 5. Topological Descriptors (Chi Descriptors)
    'Chi0', 'Chi1', 'Chi2n', 'Chi3n', 'Chi0n',
    # 6. VSA (Van der Waals Surface Area) Descriptors
    'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
    'SlogP_VSA5'
]

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
    df_combined = pd.read_csv(f'mols_gen/{str_date}/combined_molecules_with_descriptors_100K_all.csv') # tambien version 100K
    
else:
    # Read CSV files
    df = pd.read_csv(f'mols_gen/{str_date}/all_generated_molecules.csv')
    df_jtvae = pd.read_csv(f'../Data/jtvae_generated_filtered.csv')
    df_digress = pd.read_csv(f'../Data/digress_generated_filtered.csv')


    df_molecules = pd.read_csv('../Data/molecules_lt70atoms_annotated.csv')
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
    df_combined.to_csv(f'mols_gen/{str_date}/combined_molecules_with_descriptors_100K_all.csv', index=False)


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
        contador += 1

# Save the combined dataframe (optional)
df_combined.to_csv(f'mols_gen/{str_date}/combined_molecules_with_descriptors.csv', index=False)

# Define the list of descriptors to compare
columns_to_compare = DESCRIPTORS.copy() 

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
save_dir = f"mols_gen/{directories[0]}/graficas_random_10K_positive_approach"
# Save the Jensen-Shannon distances to the save directory
js_distances_kde.to_csv(os.path.join(save_dir, 'DistBetweenDists.csv'), index=False)

# Plotting function remains unchanged but uses the new descriptors
def plot_feature_distributions_with_js(dataframe, features, js_distances, bins=30):
    import random
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    sns.set(style="white")  # Change from "whitegrid" to "white"
    plt.rcParams['font.family'] = 'Nimbus Sans'

    # Define descriptive names for features
    feature_names = {
        'MolWt': 'Molecular Weight',
        'ExactMolWt': 'Exact Molecular Weight',
        'HeavyAtomCount': 'Heavy Atom Count',
        'NumValenceElectrons': 'Number of Valence Electrons',
        'NHOHCount': 'N-H/OH Count',
        'NOCount': 'N-O Count',
        'FractionCSP3': 'Fraction Csp3',
        'qed': 'Quantitative Estimate of Drug-likeness',
        'BalabanJ': "Balaban's J Index",
        'NumHDonors': 'Number of H-bond Donors',
        'NumHAcceptors': 'Number of H-bond Acceptors',
        'MolLogP': 'Molecular LogP',
        'NumRotatableBonds': 'Number of Rotatable Bonds',
        'TPSA': 'Topological Polar Surface Area',
        'NumAromaticRings': 'Number of Aromatic Rings',
        'NumAliphaticRings': 'Number of Aliphatic Rings',
        'RingCount': 'Ring Count',
        'NumSaturatedRings': 'Number of Saturated Rings',
        'BertzCT': 'Bertz Complexity',
        'MolMR': 'Molar Refractivity',
        'MaxPartialCharge': 'Maximum Partial Charge',
        'MinPartialCharge': 'Minimum Partial Charge',
        'MaxAbsPartialCharge': 'Maximum Absolute Partial Charge',
        'MinAbsPartialCharge': 'Minimum Absolute Partial Charge',
        'Ipc': 'IPC',
        'EState_VSA1': 'EState VSA Descriptor 1',
        'Chi0': 'Chi0 Index',
        'Chi1': 'Chi1 Index',
        'Chi2n': 'Chi2n Index',
        'Chi3n': 'Chi3n Index',
        'Chi0n': 'Chi0n Index',
        'SlogP_VSA1': 'SlogP VSA Descriptor 1',
        'SlogP_VSA2': 'SlogP VSA Descriptor 2',
        'SlogP_VSA3': 'SlogP VSA Descriptor 3',
        'SlogP_VSA4': 'SlogP VSA Descriptor 4',
        'SlogP_VSA5': 'SlogP VSA Descriptor 5'
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
    save_dir = f"mols_gen/{directories[0]}/graficas_random_10K_positive_approach"
    os.makedirs(save_dir, exist_ok=True)

    # Two approaches for descriptor selection
    # "Negative approach" - 60% positive, 30% mixed, 10% negative (original selection)
    negative_approach_features = [
        'MolWt',         # Tier 1, positive against both
        'MolLogP',       # Tier 1, positive against both
        'TPSA',          # Tier 1, positive against both
        'NumHAcceptors', # Tier 1, positive against both
        'FractionCSP3',  # Tier 2, positive against both
        'BalabanJ',      # Tier 2, positive against both
        'qed',           # Tier 1, positive against JTVAE, negative against Digress
        'NumAromaticRings', # Tier 3, positive against JTVAE, negative against Digress
        'MaxPartialCharge', # Tier 4, positive against JTVAE, negative against Digress
        'NumHDonors'     # Tier 1, negative against both
    ]
    
    # "Positive approach" - 70% positive, 20% mixed, 10% negative (closer to actual distribution)
    positive_approach_features = [
        'HeavyAtomCount',      # Tier 2, positive against both
        'NumValenceElectrons', # Tier 2, positive against both
        'NOCount',             # Tier 2, positive against both
        'BalabanJ',            # Tier 2, positive against both
        'NumHAcceptors',       # Tier 1, positive against both
        'RingCount',           # Tier 3, positive against both
        'TPSA',                # Tier 1, positive against both
        'qed',                 # Tier 1, positive against JTVAE, negative against Digress
        'MaxAbsPartialCharge', # Tier 4, positive against JTVAE, negative against Digress
        'NHOHCount'            # Tier 2, negative against both
    ]
    
    # SELECT YOUR APPROACH HERE:
    # Comment/uncomment one of these lines to choose the approach
    selected_features = positive_approach_features  # More positive (70/20/10)
    # selected_features = negative_approach_features  # More negative (60/30/10)

    # Create a figure with GridSpec: 3 rows and 4 columns with special layout
    fig = plt.figure(figsize=(30, 16)) # Increased size for better visualization
    
    # Create a grid with 3 rows and 4 columns
    gs = gridspec.GridSpec(3, 4)
    
    # Create axes for all 10 feature plots + log2 plot
    distribution_axes = []
    
    # First row: 4 feature plots (0,0), (0,1), (0,2), (0,3)
    for j in range(4):
        ax = fig.add_subplot(gs[0, j])
        distribution_axes.append(ax)
    
    # Second row: 3 feature plots (1,0), (1,1), (1,2)
    for j in range(3):
        ax = fig.add_subplot(gs[1, j])
        distribution_axes.append(ax)
    
    # Third row: 3 feature plots (2,0), (2,1), (2,2)
    for j in range(3):
        ax = fig.add_subplot(gs[2, j])
        distribution_axes.append(ax)
    
    # Create log2 plot spanning the right column of second and third rows
    ax_log = fig.add_subplot(gs[1:3, 3])

    # Plot distributions for each of the selected features
    for i, (ax, feature) in enumerate(zip(distribution_axes, selected_features)):
        # Add subplot letter label (A, B, C, etc.) in the upper left corner
        letter = chr(65 + i)  # 65 is ASCII for 'A'
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=28, 
                fontweight='bold', va='top', ha='left', fontname='Nimbus Sans')
        
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
                        # Special positioning for plots C, F, and H
                        if i == 2:  # Plot C (3rd plot)
                            ax.text(0.95, 0.7 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                    js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                    color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                        elif i == 5:  # Plot F (6th plot)
                            ax.text(0.95, 0.7 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                    js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                    color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                        elif i == 7:  # Plot H (8th plot)
                            ax.text(0.1, 0.8 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) - (0.075 * (cat == 'digress')), 
                                    js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                    color=style['color'], ha='left', va='center', fontname='Nimbus Sans')
                        else:
                            # Position in the right middle area, with slight vertical offsets to prevent overlap
                            ax.text(0.95, 0.3 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                    js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                    color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                    
                    # Only show the legend in the first plot (i==0)
                    show_legend = (i == 0)
                    
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
                                 ax=ax,
                                 legend=show_legend)  # Only show legend in first plot
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
            # Remove legends from all plots
            if ax.get_legend():
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
                    # Special positioning for plots C, F, and H
                    if i == 2:  # Plot C (3rd plot)
                        ax.text(0.95, 0.7 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                    elif i == 5:  # Plot F (6th plot)
                        ax.text(0.95, 0.7 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                    elif i == 7:  # Plot H (8th plot)
                        ax.text(0.1, 0.7 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) - (0.0 * (cat == 'digress')), 
                                js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                color=style['color'], ha='left', va='center', fontname='Nimbus Sans')
                    else:
                        # Position in the right middle area, with slight vertical offsets to prevent overlap
                        ax.text(0.95, 0.3 + (0.075 * (cat == 'gen_' + directories[0])) - (0.075 * (cat == 'jtvae')) + (0.0 * (cat == 'digress')), 
                                js_text, transform=ax.transAxes, fontsize=22, weight='bold',
                                color=style['color'], ha='right', va='center', fontname='Nimbus Sans')
                    
                    # Only show the legend in the first plot (i==0)
                    show_legend = (i == 0)
                
                if feature == 'Enlaces_triples':
                    if cat == reference_category:
                        sns.kdeplot(data=np.log1p(dataframe[dataframe['category'] == cat][feature]),
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=0,
                                    alpha=style['alpha'],
                                    ax=ax,
                                    legend=show_legend)
                    else:
                        sns.kdeplot(data=np.log1p(dataframe[dataframe['category'] == cat][feature]),
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=style['linewidth'],
                                    alpha=style['alpha'],
                                    clip=(np.log1p(min_val), np.log1p(max_val)),
                                    ax=ax,
                                    legend=show_legend)
                else:
                    if cat == reference_category:
                        sns.kdeplot(data=dataframe[dataframe['category'] == cat][feature],
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=0,
                                    alpha=style['alpha'],
                                    ax=ax,
                                    legend=show_legend)
                    else:
                        sns.kdeplot(data=dataframe[dataframe['category'] == cat][feature],
                                    label=label,
                                    shade=style['fill'],
                                    color=style['color'],
                                    linestyle=style['linestyle'],
                                    linewidth=style['linewidth'],
                                    alpha=style['alpha'],
                                    clip=(min_val, max_val),
                                    ax=ax,
                                    legend=show_legend)
                ax.grid(False)  # Remove grid for each subplot
                # Set tick colors to black
                ax.tick_params(axis='both', colors='black')
                for spine in ax.spines.values():
                    spine.set_color('black')
                # Set descriptive x-axis label
                ax.set_xlabel(feature_names.get(feature, feature))
            # Remove legends from all plots
            if ax.get_legend():
                ax.get_legend().remove()

    # Add legend only to the first plot
    if len(distribution_axes) > 0:
        first_ax = distribution_axes[0]
        handles, labels = first_ax.get_legend_handles_labels()
        if handles:
            first_ax.legend(handles, labels, fontsize=22)

    # Now, create the log JS ratios plot on the top-right axis (ax_log) with vertical orientation
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
    last_letter = chr(65 + len(distribution_axes))  # After all distribution plots
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

# New code: Compute log2 ratios for descriptors and save to CSV
# Define tier mapping based on manual classification

tier_dict = {
    'MolWt': 'Tier 1',
    'MolLogP': 'Tier 1',
    'TPSA': 'Tier 1',
    'NumHDonors': 'Tier 1',
    'NumHAcceptors': 'Tier 1',
    'qed': 'Tier 1',
    'ExactMolWt': 'Tier 2',
    'HeavyAtomCount': 'Tier 2',
    'NumValenceElectrons': 'Tier 2',
    'FractionCSP3': 'Tier 2',
    'BalabanJ': 'Tier 2',
    'NHOHCount': 'Tier 2',
    'NOCount': 'Tier 2',
    'NumRotatableBonds': 'Tier 3',
    'NumAromaticRings': 'Tier 3',
    'NumAliphaticRings': 'Tier 3',
    'RingCount': 'Tier 3',
    'NumSaturatedRings': 'Tier 3',
    'BertzCT': 'Tier 3',
    'MolMR': 'Tier 4',
    'MaxPartialCharge': 'Tier 4',
    'MinPartialCharge': 'Tier 4',
    'MaxAbsPartialCharge': 'Tier 4',
    'MinAbsPartialCharge': 'Tier 4',
    'Ipc': 'Tier 4',
    'EState_VSA1': 'Tier 4',
    'Chi0': 'Tier 5',
    'Chi1': 'Tier 5',
    'Chi2n': 'Tier 5',
    'Chi3n': 'Tier 5',
    'Chi0n': 'Tier 5',
    'SlogP_VSA1': 'Tier 5',
    'SlogP_VSA2': 'Tier 5',
    'SlogP_VSA3': 'Tier 5',
    'SlogP_VSA4': 'Tier 5',
    'SlogP_VSA5': 'Tier 5'
}

epsilon = 0.0001
result_list = []

# Iterate over each descriptor in columns_to_compare (which is a copy of DESCRIPTORS)
for feat in columns_to_compare:
    # Query for our model JS distance: comparing original ('ori') to our model ('gen_' + directories[0])
    row_ours = js_distances_kde[(js_distances_kde['Feature'] == feat) &
                (js_distances_kde['Category1'] == 'ori') &
                (js_distances_kde['Category2'] == 'gen_' + directories[0])]
    # Query for jtvae: comparing original ('ori') with 'jtvae'
    row_jtvae = js_distances_kde[(js_distances_kde['Feature'] == feat) &
                (js_distances_kde['Category1'] == 'ori') &
                (js_distances_kde['Category2'] == 'jtvae')]
    # Query for digress: comparing original ('ori') with 'digress'
    row_digress = js_distances_kde[(js_distances_kde['Feature'] == feat) &
                (js_distances_kde['Category1'] == 'ori') &
                (js_distances_kde['Category2'] == 'digress')]

    js_ours = row_ours.iloc[0]['JS_Distance'] if not row_ours.empty else np.nan
    js_jtvae = row_jtvae.iloc[0]['JS_Distance'] if not row_jtvae.empty else np.nan
    js_digress = row_digress.iloc[0]['JS_Distance'] if not row_digress.empty else np.nan

    # Replace zero values with epsilon to avoid division issues
    if js_ours == 0:
        js_ours = epsilon
    if js_jtvae == 0:
        js_jtvae = epsilon
    if js_digress == 0:
        js_digress = epsilon

    # Compute log2 ratios if values are available
    log2_ratio_jtvae = np.log2(js_jtvae / js_ours) if (not np.isnan(js_ours) and not np.isnan(js_jtvae)) else np.nan
    log2_ratio_digress = np.log2(js_digress / js_ours) if (not np.isnan(js_ours) and not np.isnan(js_digress)) else np.nan

    result_list.append({
        'Descriptor': feat,
        'Tier': tier_dict.get(feat, 'Unknown'),
        'log2_ratio_jtvae': log2_ratio_jtvae,
        'log2_ratio_digress': log2_ratio_digress
    })

result_df = pd.DataFrame(result_list)
output_csv = os.path.join(save_dir, 'descriptor_log2_ratios.csv')
result_df.to_csv(output_csv, index=False)
print(f"Descriptor log2 ratios CSV saved to {output_csv}")
