# To generate the anionic forms of epicatechin, typically you would deprotonate 
# one or more of the phenolic OH groups, resulting in phenolate (O–) anions. 
# This is common in alkaline conditions or during certain reactions.

# Python + RDKit Approach:
# You can use RDKit to programmatically generate these anionic forms by modifying the hydrogen atoms on the phenolic oxygens.

# Here's an example workflow:

# Parse the neutral molecule from SMILES.
# Identify phenolic -OH groups.
# Remove a hydrogen to make an O– anion.
# Generate SMILES for each anionic form.

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import itertools

def generate_epicatechin_anions():
    # Original epicatechin SMILES (with explicit Hs for accuracy)
    smiles = "C1[C@H]([C@@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)O"
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Failed to parse SMILES string")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # More precise identification of phenolic OH groups
    phenol_smarts = "[OH]-[c]"
    phenol_pattern = Chem.MolFromSmarts(phenol_smarts)
    phenol_matches = mol.GetSubstructMatches(phenol_pattern)
    
    phenol_oxys = []
    for match in phenol_matches:
        o_idx = match[0]  # Oxygen atom index
        h_idx = None
        
        # Find the hydrogen attached to this oxygen
        oxygen = mol.GetAtomWithIdx(o_idx)
        for neighbor in oxygen.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # Hydrogen
                h_idx = neighbor.GetIdx()
                break
        
        if h_idx is not None:
            phenol_oxys.append((o_idx, h_idx))
    
    n_sites = len(phenol_oxys)
    print(f"Found {n_sites} phenolic OH groups.")
    
    anionic_smiles_list = []
    
    # Generate all possible deprotonation combinations
    for r in range(1, n_sites + 1):
        for combo in itertools.combinations(range(n_sites), r):
            mol_copy = Chem.RWMol(mol)
            
            # Collect hydrogen indices to remove (in reverse order)
            to_remove = [phenol_oxys[idx][1] for idx in combo]
            to_remove.sort(reverse=True)
            
            # Remove hydrogens
            for hyd_idx in to_remove:
                mol_copy.RemoveAtom(hyd_idx)
            
            # Set formal charge on oxygen atoms
            for idx in combo:
                oxy_idx = phenol_oxys[idx][0]
                atom = mol_copy.GetAtomWithIdx(oxy_idx)
                atom.SetFormalCharge(-1)
            
            # Generate canonical SMILES
            try:
                anionic_smiles = Chem.MolToSmiles(mol_copy, canonical=True)
                anionic_smiles_list.append(anionic_smiles)
            except Exception as e:
                print(f"Error generating SMILES: {e}")
    
    print("Generated anionic SMILES:")
    for i, s in enumerate(anionic_smiles_list, 1):
        print(f"{i}: {s}")
    
    # Generate images
    for i, s in enumerate(anionic_smiles_list, 1):
        try:
            mol_img = Chem.MolFromSmiles(s)
            if mol_img:
                # Generate 2D coordinates for better visualization
                AllChem.Compute2DCoords(mol_img)
                img = Draw.MolToImage(mol_img, size=(300, 300))
                filename = f"epicatechin_anion_{i}.png"
                img.save(filename)
                print(f"Saved: {filename}")
            else:
                print(f"Could not parse SMILES: {s}")
        except Exception as e:
            print(f"Error generating image for {s}: {e}")
    
    return anionic_smiles_list

# Run the function
if __name__ == "__main__":
    anions = generate_epicatechin_anions()

#  Key Improvements:

# More precise phenolic OH identification using SMARTS pattern matching

# Added error handling for molecule creation and SMILES generation

# Preserved stereochemistry in the original SMILES string

# Added 2D coordinate generation for better visualization

# Better code organization with a main function

# More informative error messages

# Expected Output:
# The code should identify 4 phenolic OH groups in epicatechin and generate:

# 4 mono-anions

# 6 di-anions

# 4 tri-anions

# 1 tetra-anion

# For a total of 15 anionic structures, which will be saved as PNG files with names like "epicatechin_anion_1.png", "epicatechin_anion_2.png", etc.

