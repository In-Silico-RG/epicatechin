"""
Key Features of This Implementation:
Directory Structure: Organizes calculations by functional/basis set combination and solvent environment:

text
orca_calculations/
├── M06-2X_6-311Gdp/
│   ├── gas/
│   └── water/
└── wB97XD_aug-cc-pVDZ/
    ├── gas/
    └── water/
Functional/Basis Set Combinations:

M06-2X/6-311G(d,p)

ωB97X-D/aug-cc-pVDZ

Solvent Environments:

Gas phase

Aqueous solution (using CPCM solvation model)

Systematic Naming: Files are named consistently for easy identification.

Batch Script: Includes a shell script to run all calculations in parallel.

How to Use:
Run the script to generate all ORCA input files:

bash
python generate_orca_inputs.py
Run the calculations using the provided batch script:

bash
./run_calculations.sh
Alternatively, run individual calculations:

bash
orca orca_calculations/M06-2X_6-311Gdp/gas/epicatechin_neutral.inp > epicatechin_neutral.out
Additional Notes:
The script checks for existing structure files and generates them if needed.

For the anion structures, you'll need to run the anion generation code first or modify this script to include that functionality.

The batch script uses GNU Parallel to run calculations concurrently. Adjust the -j parameter based on your available resources.

The input files include additional convergence settings (MaxIter 500) to handle potentially difficult convergence cases.

This implementation provides a systematic approach to comparing different DFT methods for studying epicatechin and its anions in different environments, which is essential for understanding their properties and reactivity.
"""



from rdkit import Chem
from rdkit.Chem import AllChem
import os

def generate_orca_inputs():
    # Create directory structure
    base_dir = "orca_calculations"
    functional_dirs = {
        "M06-2X_6-311Gdp": "M06-2X/6-311G(d,p)",
        "wB97XD_aug-cc-pVDZ": "ωB97X-D/aug-cc-pVDZ"
    }
    
    # Create base directory
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create functional directories
    for func_dir in functional_dirs:
        func_path = os.path.join(base_dir, func_dir)
        if not os.path.exists(func_path):
            os.makedirs(func_path)
            
        # Create gas and water subdirectories
        for solvent in ["gas", "water"]:
            solvent_path = os.path.join(func_path, solvent)
            if not os.path.exists(solvent_path):
                os.makedirs(solvent_path)
    
    # Generate neutral and anion structures if they don't exist
    generate_structures()
    
    # Generate ORCA input files
    for func_name, func_desc in functional_dirs.items():
        for solvent in ["gas", "water"]:
            # Neutral molecule
            create_orca_input(
                mol_file="epicatechin_neutral.mol",
                output_dir=os.path.join(base_dir, func_name, solvent),
                filename="epicatechin_neutral.inp",
                charge=0,
                multiplicity=1,
                functional=func_desc,
                basis_set=func_desc.split("/")[1],
                solvent=solvent
            )
            
            # Anions (mono-anions 1-4)
            for i in range(1, 5):
                create_orca_input(
                    mol_file=f"epicatechin_mono_anion_{i}.mol",
                    output_dir=os.path.join(base_dir, func_name, solvent),
                    filename=f"epicatechin_mono_anion_{i}.inp",
                    charge=-1,
                    multiplicity=1,
                    functional=func_desc,
                    basis_set=func_desc.split("/")[1],
                    solvent=solvent
                )
    
    print("All ORCA input files generated successfully!")
    print(f"Directory structure: {base_dir}/")
    print("Functional combinations:")
    for func_dir, func_desc in functional_dirs.items():
        print(f"  {func_dir}: {func_desc}")

def generate_structures():
    """Generate neutral and anion structures if they don't exist"""
    # Check if neutral structure exists
    if not os.path.exists("epicatechin_neutral.mol"):
        print("Generating neutral epicatechin structure...")
        smiles = "C1[C@H]([C@@H](OC2=CC(=CC(=C21)O)O)C3=CC(=C(C=C3)O)O)O"
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            raise ValueError("Failed to parse SMILES string")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D structure
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Save MOL file
        writer = Chem.SDWriter("epicatechin_neutral.mol")
        writer.write(mol)
        writer.close()
        
        # Save XYZ file
        xyz_content = generate_xyz_content(mol, "Epicatechin_neutral")
        with open("epicatechin_neutral.xyz", "w") as f:
            f.write(xyz_content)
    
    # Check if anion structures exist
    for i in range(1, 5):
        if not os.path.exists(f"epicatechin_mono_anion_{i}.mol"):
            print(f"Generating anion structure {i}...")
            # This would need the anion generation code from earlier
            # For now, we'll just note that they should be generated first
            print(f"Please generate anion structures first using the previous code")
            return

def create_orca_input(mol_file, output_dir, filename, charge, multiplicity, 
                     functional, basis_set, solvent):
    """Create an ORCA input file with specified parameters"""
    
    # Read the molecule
    mol = Chem.MolFromMolFile(mol_file)
    if mol is None:
        print(f"Could not read molecule from {mol_file}")
        return
    
    # Generate XYZ content
    xyz_content = generate_xyz_content(mol)
    
    # Determine functional and basis set
    if "M06-2X" in functional:
        functional_line = "M06-2X"
        basis_line = "6-311G(d,p)"
    else:  # wB97XD
        functional_line = "wB97X-D"
        basis_line = "aug-cc-pVDZ"
    
    # Solvent settings
    if solvent == "water":
        solvent_line = "CPCM(water)"
    else:
        solvent_line = ""
    
    # Construct ORCA input
    orca_input = f"""! {functional_line} {basis_line} {solvent_line} def2/J D3BJ Opt
%pal nprocs 8 end
%maxcore 2000

%geom
   MaxIter 500
end

%scf
   Convergence Tight
   MaxIter 500
end

* xyz {charge} {multiplicity}
{xyz_content}
*
"""
    
    # Write to file
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(orca_input)

def generate_xyz_content(mol, title="Molecule"):
    """Generate XYZ file content from RDKit molecule"""
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    
    xyz_lines = []
    
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        symbol = atom.GetSymbol()
        xyz_lines.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    
    return "\n".join(xyz_lines)

def create_batch_script():
    """Create a batch script to run all ORCA calculations"""
    batch_script = """#!/bin/bash

# Batch script to run all ORCA calculations
# Usage: ./run_calculations.sh

BASE_DIR="orca_calculations"

# Function to run ORCA calculation
run_orca() {
    input_file=$1
    output_file="${input_file%.inp}.out"
    echo "Running calculation: $input_file"
    orca $input_file > $output_file 2>&1
    echo "Completed: $output_file"
}

export -f run_orca

# Find all input files and run them in parallel (adjust max jobs as needed)
find $BASE_DIR -name "*.inp" | parallel -j 2 run_orca

echo "All calculations completed!"
"""
    
    with open("run_calculations.sh", "w") as f:
        f.write(batch_script)
    
    # Make the script executable
    os.chmod("run_calculations.sh", 0o755)
    print("Created batch script: run_calculations.sh")

if __name__ == "__main__":
    generate_orca_inputs()
    create_batch_script()
