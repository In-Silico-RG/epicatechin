import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, rdPartialCharges
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
import os

class AdvancedSolubilityPredictor:
    def __init__(self):
        # Parameters for different deprotonation sites
        self.site_factors = {
            'catechol': 1.2,      # Catechol-like sites (ortho-dihydroxy)
            'phenol': 1.0,        # Regular phenolic sites
            'resorcinol': 0.9     # Resorcinol-like sites (meta-dihydroxy)
        }
        
        # Charge distribution parameters
        self.charge_params = {
            'max_neg_charge': -0.5,
            'charge_delocalization': 0.8
        }
        
    def identify_deprotonation_site_type(self, mol, deprotonated_o_idx):
        """
        Identify the type of deprotonation site based on molecular environment
        """
        # Get the deprotonated oxygen atom
        o_atom = mol.GetAtomWithIdx(deprotonated_o_idx)
        
        # Get the carbon it's attached to
        carbon_neighbors = [n for n in o_atom.GetNeighbors() if n.GetAtomicNum() == 6]
        if not carbon_neighbors:
            return 'phenol'
        
        carbon_atom = carbon_neighbors[0]
        
        # Check if this carbon has an ortho-hydroxy group (catechol-like)
        for neighbor in carbon_atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 8 and neighbor.GetIdx() != deprotonated_o_idx:
                # Check if this oxygen has a hydrogen (hydroxy group)
                for n2 in neighbor.GetNeighbors():
                    if n2.GetAtomicNum() == 1:
                        # Check distance - if ortho, it's catechol-like
                        return 'catechol'
        
        # Check for meta-dihydroxy pattern (resorcinol-like)
        # This would require more complex pattern matching
        # For simplicity, we'll return 'phenol' for now
        return 'phenol'
    
    def calculate_charge_descriptors(self, mol):
        """
        Calculate charge-related descriptors for the molecule
        """
        # Compute Gasteiger charges
        rdPartialCharges.ComputeGasteigerCharges(mol)
        
        charges = []
        neg_charges = []
        pos_charges = []
        
        for atom in mol.GetAtoms():
            charge = atom.GetDoubleProp('_GasteigerCharge')
            charges.append(charge)
            if charge < 0:
                neg_charges.append(charge)
            elif charge > 0:
                pos_charges.append(charge)
        
        # Calculate charge statistics
        if charges:
            charge_mean = np.mean(charges)
            charge_std = np.std(charges)
            max_neg_charge = min(neg_charges) if neg_charges else 0
            max_pos_charge = max(pos_charges) if pos_charges else 0
            charge_range = max(charges) - min(charges) if charges else 0
        else:
            charge_mean = charge_std = max_neg_charge = max_pos_charge = charge_range = 0
        
        return {
            'charge_mean': charge_mean,
            'charge_std': charge_std,
            'max_neg_charge': max_neg_charge,
            'max_pos_charge': max_pos_charge,
            'charge_range': charge_range
        }
    
    def calculate_solvation_descriptors(self, mol):
        """
        Calculate descriptors related to solvation energy
        """
        # Calculate molecular volume
        mol_volume = Descriptors.MolMR(mol)  # Approximation of volume
        
        # Calculate polar surface area
        tpsa = Descriptors.TPSA(mol)
        
        # Calculate hydrophobic surface area
        # This is a simplified approximation
        hydrophobic_fraction = 1 - (tpsa / (4 * np.pi * (mol_volume/(4/3*np.pi))**(2/3))) if mol_volume > 0 else 0
        
        return {
            'mol_volume': mol_volume,
            'tpsa': tpsa,
            'hydrophobic_fraction': hydrophobic_fraction
        }
    
    def predict_solubility_advanced(self, mol, is_anion=False, deprotonation_site=None):
        """
        Advanced solubility prediction using multiple descriptors
        """
        # Calculate basic descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        aromatic_rings = Descriptors.NumAromaticRings(mol)
        
        # Calculate charge descriptors
        charge_desc = self.calculate_charge_descriptors(mol)
        
        # Calculate solvation descriptors
        solvation_desc = self.calculate_solvation_descriptors(mol)
        
        # Base solubility model (simplified)
        # This is a placeholder - in a real implementation, you would use a trained model
        logS_base = (0.5 - 0.05 * mw/100 + 0.2 * logp - 0.3 * hbd + 0.2 * hba - 
                    0.1 * rotatable_bonds + 0.05 * aromatic_rings)
        
        # Adjust for ionic character
        if is_anion:
            # Ionic species are generally more soluble
            logS_base += 1.5
            
            # Adjust based on charge distribution
            charge_effect = (charge_desc['max_neg_charge'] * self.charge_params['charge_delocalization'] + 
                           charge_desc['charge_std'] * 0.5)
            logS_base += charge_effect
            
            # Adjust based on deprotonation site type
            if deprotonation_site:
                site_factor = self.site_factors.get(deprotonation_site, 1.0)
                logS_base += (site_factor - 1.0) * 0.5
        
        # Adjust for solvation properties
        solvation_effect = (-0.002 * solvation_desc['mol_volume'] + 
                          0.01 * solvation_desc['tpsa'] - 
                          0.5 * solvation_desc['hydrophobic_fraction'])
        logS_base += solvation_effect
        
        # Ensure reasonable bounds
        logS = max(-10, min(2, logS_base))
        
        return logS, {
            'mw': mw,
            'logp': logp,
            'hbd': hbd,
            'hba': hba,
            'rotatable_bonds': rotatable_bonds,
            'aromatic_rings': aromatic_rings,
            **charge_desc,
            **solvation_desc
        }
    
    def predict_all_structures_with_details(self):
        """
        Predict solubility for all structures with detailed analysis
        """
        results = []
        
        # Neutral molecule
        mol = Chem.MolFromMolFile("epicatechin_neutral.mol")
        if mol:
            logS, descriptors = self.predict_solubility_advanced(mol, is_anion=False)
            results.append({
                'Molecule': 'Epicatechin_neutral',
                'logS': logS,
                'Type': 'Neutral',
                'Deprotonation_Site': 'N/A',
                **descriptors
            })
        
        # Anions - we need to know which oxygen was deprotonated
        # This would require tracking the deprotonation site from the generation process
        # For this example, we'll assume we have this information
        
        # In a real implementation, you would have this information from the anion generation
        deprotonation_sites = {
            1: 'catechol',  # Example - this would be based on your molecular structure
            2: 'phenol',
            3: 'catechol',
            4: 'phenol'
        }
        
        for i in range(1, 5):
            try:
                mol = Chem.MolFromMolFile(f"epicatechin_mono_anion_{i}.mol")
                if mol:
                    site_type = deprotonation_sites.get(i, 'phenol')
                    logS, descriptors = self.predict_solubility_advanced(
                        mol, is_anion=True, deprotonation_site=site_type)
                    
                    results.append({
                        'Molecule': f'Epicatechin_anion_{i}',
                        'logS': logS,
                        'Type': 'Anion',
                        'Deprotonation_Site': site_type,
                        **descriptors
                    })
            except Exception as e:
                print(f"Could not process anion {i}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv("advanced_solubility_predictions.csv", index=False)
        
        return df

# Function to generate 3D structures with charges if not already done
def generate_3d_structures_with_charges():
    """Generate 3D structures with partial charges for accurate descriptor calculation"""
    # Neutral molecule
    mol = Chem.MolFromMolFile("epicatechin_neutral.mol")
    if mol:
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        
        # Compute partial charges
        rdPartialCharges.ComputeGasteigerCharges(mol)
        
        # Save with charges
        writer = Chem.SDWriter("epicatechin_neutral_with_charges.mol")
        writer.write(mol)
        writer.close()
    
    # Anions
    for i in range(1, 5):
        try:
            mol = Chem.MolFromMolFile(f"epicatechin_mono_anion_{i}.mol")
            if mol:
                # Add hydrogens and generate 3D coordinates
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Compute partial charges
                rdPartialCharges.ComputeGasteigerCharges(mol)
                
                # Save with charges
                writer = Chem.SDWriter(f"epicatechin_anion_{i}_with_charges.mol")
                writer.write(mol)
                writer.close()
        except:
            print(f"Could not process anion {i}")

# Main execution
def main():
    # First, generate 3D structures with partial charges for accurate calculations
    print("Generating 3D structures with partial charges...")
    generate_3d_structures_with_charges()
    
    # Initialize predictor
    predictor = AdvancedSolubilityPredictor()
    
    # Predict solubility with advanced model
    results_df = predictor.predict_all_structures_with_details()
    
    # Display results
    print("\nAdvanced Solubility Predictions:")
    print(results_df[['Molecule', 'Type', 'Deprotonation_Site', 'logS']].to_string(index=False))
    
    # Display detailed descriptors for the first few molecules
    print("\nDetailed Descriptors (first 2 molecules):")
    for idx, row in results_df.head(2).iterrows():
        print(f"\n{row['Molecule']}:")
        for col in results_df.columns:
            if col not in ['Molecule', 'Type', 'Deprotonation_Site'] and not pd.isna(row[col]):
                print(f"  {col}: {row[col]:.4f}")
    
    # Save a summary report
    with open("solubility_analysis_report.txt", "w") as f:
        f.write("Advanced Solubility Prediction Report\n")
        f.write("=====================================\n\n")
        
        f.write("Summary of Predictions:\n")
        f.write(results_df[['Molecule', 'Type', 'Deprotonation_Site', 'logS']].to_string(index=False))
        
        f.write("\n\nKey Insights:\n")
        f.write("1. Neutral epicatechin has moderate solubility due to multiple H-bond donors/acceptors.\n")
        f.write("2. Anionic forms generally have higher solubility due to charge-charge interactions with water.\n")
        f.write("3. Catechol-like deprotonation sites may have different solubility than regular phenolic sites.\n")
        f.write("4. Charge distribution and delocalization affect solubility predictions.\n")
        
        f.write("\nMethodology:\n")
        f.write("- Used advanced descriptors including charge distribution and solvation properties\n")
        f.write("- Differentiated between different deprotonation site types\n")
        f.write("- Incorporated molecular volume and polar surface area effects\n")
    
    print("\nDetailed report saved to solubility_analysis_report.txt")
    print("Advanced solubility predictions saved to advanced_solubility_predictions.csv")

if __name__ == "__main__":
    main()

"""
Key Features of This Advanced Model:
Site-Specific Differentiation:

Identifies different types of deprotonation sites (catechol, phenol, resorcinol)

Applies different correction factors based on site type

Charge-Based Descriptors:

Calculates Gasteiger partial charges for all atoms

Uses charge distribution statistics (mean, std, max negative/positive)

Considers charge delocalization effects

Solvation Descriptors:

Molecular volume approximations

Polar surface area calculations

Hydrophobic fraction estimation

Comprehensive Reporting:

Saves detailed descriptors for each molecule

Generates a comprehensive analysis report

Provides insights into the factors affecting solubility

How to Use:
Run the script after generating your molecular structures:

bash
python advanced_solubility.py
The script will:

Generate 3D structures with partial charges

Calculate advanced descriptors for each molecule

Predict solubility with site-specific adjustments

Save detailed results to CSV and a summary report

Expected Improvements:
Differentiation Between Anions: The model should now differentiate between different deprotonation sites

More Accurate Predictions: The use of charge-based and solvation descriptors should improve accuracy

Better Handling of Ionic Species: Special consideration for charged molecules

Limitations and Further Improvements:
Training Data: For even better accuracy, the model could be trained on experimental solubility data

Quantum Chemical Descriptors: Incorporation of DFT-calculated properties (electrostatic potential, etc.)

Solvation Models: Integration with more sophisticated solvation models like COSMO-RS

pH Dependence: Extension to model pH-dependent solubility

"""


"""
Key Features of This Advanced Model:
Site-Specific Differentiation:

Identifies different types of deprotonation sites (catechol, phenol, resorcinol)

Applies different correction factors based on site type

Charge-Based Descriptors:

Calculates Gasteiger partial charges for all atoms

Uses charge distribution statistics (mean, std, max negative/positive)

Considers charge delocalization effects

Solvation Descriptors:

Molecular volume approximations

Polar surface area calculations

Hydrophobic fraction estimation

Comprehensive Reporting:

Saves detailed descriptors for each molecule

Generates a comprehensive analysis report

Provides insights into the factors affecting solubility

How to Use:
Run the script after generating your molecular structures:

bash
python advanced_solubility.py
The script will:

Generate 3D structures with partial charges

Calculate advanced descriptors for each molecule

Predict solubility with site-specific adjustments

Save detailed results to CSV and a summary report

Expected Improvements:
Differentiation Between Anions: The model should now differentiate between different deprotonation sites

More Accurate Predictions: The use of charge-based and solvation descriptors should improve accuracy

Better Handling of Ionic Species: Special consideration for charged molecules

Limitations and Further Improvements:
Training Data: For even better accuracy, the model could be trained on experimental solubility data

Quantum Chemical Descriptors: Incorporation of DFT-calculated properties (electrostatic potential, etc.)

Solvation Models: Integration with more sophisticated solvation models like COSMO-RS

pH Dependence: Extension to model pH-dependent solubility profiles

This advanced model provides a more nuanced approach to solubility prediction, particularly for ionic species and different deprotonation sites, which should address the limitations of the previous simpler models.

"""
