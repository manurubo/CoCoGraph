import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import random
import os

# Path to the CSV file
csv_path = "mols_gen/250211_database_allmolecules_main_2_22_confps_timepred_2_22_confps_sinexplicit/all_generated_molecules.csv"

# Set non-GUI backend to avoid hanging
import matplotlib
matplotlib.use('Agg')

print("Reading CSV file...")
# Read the CSV file
df = pd.read_csv(csv_path)
print(f"Found {len(df)} molecules in the CSV file")

# Ensure we have the required columns
if 'smiles' not in df.columns or 'molecular_formula' not in df.columns:
    raise ValueError("CSV file must contain 'smiles' and 'molecular_formula' columns")

# Randomly select 50 molecules
if len(df) < 50:
    print(f"Warning: CSV contains only {len(df)} molecules, using all available.")
    selected_molecules = df
else:
    print("Randomly selecting 50 molecules...")
    selected_molecules = df.sample(50, random_state=5335748)

# Create mols and legends lists
mols = []
legends = []

print("Processing molecules...")
for i, (_, row) in enumerate(selected_molecules.iterrows()):
    if len(mols) >= 50:
        break
    
    try:
        smiles = row['smiles']
        formula = row['molecular_formula']
        
        # Create RDKit molecule from SMILES
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print(f"Warning: Could not parse SMILES: {smiles}")
            continue
        
        # Add the molecule and its formula to our lists
        mols.append(mol)
        legends.append(formula)
        
    except Exception as e:
        print(f"Error processing molecule {i}: {e}")
        continue

print(f"Successfully processed {len(mols)} molecules")

# If we couldn't get enough valid molecules
if len(mols) < 50:
    print(f"Warning: Only found {len(mols)} valid molecules instead of 50")

# Use RDKit's built-in grid image generator
print("Generating grid image...")
img = Draw.MolsToGridImage(
    mols, 
    molsPerRow=5,
    subImgSize=(300, 200),
    legends=legends,
    useSVG=False
)

# Save the image
output_path = "molecule_grid.png"
img.save(output_path)
print(f"Saved molecule grid to {os.path.abspath(output_path)}") 