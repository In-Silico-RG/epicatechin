# Key Features of This Implementation:
# Focuses on Mono-Anions: Only generates structures with a single deprotonation.

# 3D Structure Generation: Uses RDKit's ETKDG method to generate 3D coordinates and MMFF force field optimization.
# Multiple File Formats:
# PNG: 2D visualization
# MOL: Standard chemical format with 3D coordinates
# XYZ: Simple text format with atomic coordinates
# Improved Accuracy:
# Uses stereochemistry-aware SMILES
# Properly identifies phenolic OH groups with SMARTS pattern
# Generates 3D-optimized structures
# Expected Output:
# The code will:
# Identify the phenolic OH groups in epicatechin (should find 4)
# Generate 4 mono-anions (one for each possible deprotonation site)
# Save each anion in three formats:
# epicatechin_mono_anion_1.png, epicatechin_mono_anion_2.png, etc.
# epicatechin_mono_anion_1.mol, epicatechin_mono_anion_2.mol, etc.
# epicatechin_mono_anion_1.xyz, epicatechin_mono_anion_2.xyz, etc.
# The XYZ files will contain the atomic coordinates in a standard format that can be used for further computational chemistry calculations or visualization in other software.

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import itertools

def generate_epicatechin_mono_anions():
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
    
    # Generate only mono-anions (single deprotonation)
    anionic_mols = []
    
    for i in range(n_sites):
        mol_copy = Chem.RWMol(mol)
        
        # Remove the hydrogen
        h_idx = phenol_oxys[i][1]
        mol_copy.RemoveAtom(h_idx)
        
        # Set formal charge on oxygen atom
        oxy_idx = phenol_oxys[i][0]
        atom = mol_copy.GetAtomWithIdx(oxy_idx)
        atom.SetFormalCharge(-1)
        
        # Generate 3D structure
        mol_3d = Chem.Mol(mol_copy)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol_3d)
        
        anionic_mols.append(mol_3d)
    
    # Save files for each mono-anion
    for i, mol_anion in enumerate(anionic_mols, 1):
        # Generate canonical SMILES
        anionic_smiles = Chem.MolToSmiles(mol_anion, canonical=True)
        print(f"Mono-anion {i}: {anionic_smiles}")
        
        # Save PNG image
        img = Draw.MolToImage(mol_anion, size=(300, 300))
        img.save(f"epicatechin_mono_anion_{i}.png")
        
        # Save MOL file
        writer = Chem.SDWriter(f"epicatechin_mono_anion_{i}.mol")
        writer.write(mol_anion)
        writer.close()
        
        # Save XYZ file
        xyz_content = generate_xyz_content(mol_anion, f"Epicatechin_mono_anion_{i}")
        with open(f"epicatechin_mono_anion_{i}.xyz", "w") as f:
            f.write(xyz_content)
        
        print(f"Saved files for mono-anion {i}: PNG, MOL, and XYZ formats")
    
    return anionic_mols

def generate_xyz_content(mol, title="Molecule"):
    """Generate XYZ file content from RDKit molecule"""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    
    xyz_lines = [str(num_atoms), title]
    
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        xyz_lines.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    
    return "\n".join(xyz_lines)

# Run the function
if __name__ == "__main__":
    anions = generate_epicatechin_mono_anions()
    print("All mono-anions generated and saved successfully!")
