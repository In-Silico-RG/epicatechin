import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf, rdPartialCharges, rdMolDescriptors
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig
import os

class AdvancedMolecularDescriptorCalculator:
    def __init__(self):
        # Use the specific path you provided for BaseFeatures.fdef
        self.custom_fdef_path = "/home/aldo/SOFT/rdkit_bak/Data/BaseFeatures.fdef"
        self.feature_factory = None
        
        try:
            # First try the custom path
            if os.path.exists(self.custom_fdef_path):
                self.feature_factory = ChemicalFeatures.BuildFeatureFactory(self.custom_fdef_path)
                print(f"Successfully loaded BaseFeatures.fdef from custom path: {self.custom_fdef_path}")
            else:
                # Fall back to RDDataDir
                fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                if os.path.exists(fdefName):
                    self.feature_factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
                    print("Successfully loaded BaseFeatures.fdef from RDDataDir")
                else:
                    print("Warning: BaseFeatures.fdef not found. Pharmacophore features will be limited.")
        except Exception as e:
            print(f"Warning: Could not initialize feature factory: {e}. Pharmacophore features will be limited.")
    
    def calculate_3d_descriptors(self, mol):
        """Calculate 3D molecular descriptors"""
        if not mol.GetNumConformers():
            # Generate 3D structure if not present
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        
        descriptors = {}
        
        # Calculate radius of gyration
        try:
            descriptors['radius_of_gyration'] = rdMolDescriptors.CalcRadiusOfGyration(mol)
        except:
            descriptors['radius_of_gyration'] = 0
        
        # Calculate principal moments of inertia
        try:
            moments = rdMolDescriptors.CalcPrincipalMomentsOfInertia(mol)
            descriptors['pmi1'] = moments[0]
            descriptors['pmi2'] = moments[1]
            descriptors['pmi3'] = moments[2]
            descriptors['pmi_ratio'] = moments[0] / moments[2] if moments[2] > 0 else 0
        except:
            descriptors.update({'pmi1': 0, 'pmi2': 0, 'pmi3': 0, 'pmi_ratio': 0})
        
        # Calculate asphericity
        try:
            descriptors['asphericity'] = rdMolDescriptors.CalcAsphericity(mol)
        except:
            descriptors['asphericity'] = 0
            
        # Calculate eccentricity
        try:
            descriptors['eccentricity'] = rdMolDescriptors.CalcEccentricity(mol)
        except:
            descriptors['eccentricity'] = 0
            
        return descriptors
    
    def calculate_electronic_descriptors(self, mol):
        """Calculate electronic descriptors"""
        # Compute Gasteiger charges if not already done
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except:
            pass
            
        descriptors = {}
        charges = []
        neg_charges = []
        pos_charges = []
        oxygen_charges = []
        
        for atom in mol.GetAtoms():
            try:
                charge = atom.GetDoubleProp('_GasteigerCharge')
                charges.append(charge)
                
                if charge < 0:
                    neg_charges.append(charge)
                elif charge > 0:
                    pos_charges.append(charge)
                    
                if atom.GetAtomicNum() == 8:  # Oxygen
                    oxygen_charges.append(charge)
            except:
                pass
        
        # Calculate charge statistics
        if charges:
            descriptors['charge_mean'] = np.mean(charges)
            descriptors['charge_std'] = np.std(charges)
            descriptors['max_neg_charge'] = min(neg_charges) if neg_charges else 0
            descriptors['max_pos_charge'] = max(pos_charges) if pos_charges else 0
            descriptors['charge_range'] = max(charges) - min(charges) if charges else 0
        else:
            descriptors.update({
                'charge_mean': 0, 'charge_std': 0, 'max_neg_charge': 0, 
                'max_pos_charge': 0, 'charge_range': 0
            })
        
        # Oxygen charge statistics
        if oxygen_charges:
            descriptors['oxygen_charge_mean'] = np.mean(oxygen_charges)
            descriptors['oxygen_charge_std'] = np.std(oxygen_charges)
            descriptors['most_neg_oxygen'] = min(oxygen_charges)
        else:
            descriptors.update({
                'oxygen_charge_mean': 0, 'oxygen_charge_std': 0, 'most_neg_oxygen': 0
            })
            
        return descriptors
    
    def calculate_basic_pharmacophore_features(self, mol):
        """Calculate basic pharmacophore features without requiring fdef file"""
        descriptors = {}
        
        # Count donors and acceptors using basic RDKit functions
        descriptors['Donor'] = Lipinski.NumHDonors(mol)
        descriptors['Acceptor'] = Lipinski.NumHAcceptors(mol)
        
        # Count ionizable groups
        neg_ionizable = 0
        pos_ionizable = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == -1:  # Negatively charged oxygen
                neg_ionizable += 1
            elif atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:  # Positively charged nitrogen
                pos_ionizable += 1
        
        descriptors['NegIonizable'] = neg_ionizable
        descriptors['PosIonizable'] = pos_ionizable
        
        # Estimate hydrophobic features based on carbon atoms without polar neighbors
        hydrophobic_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6:  # Carbon
                # Check if it has no polar neighbors (O, N, S, P, F, Cl, Br, I)
                has_polar_neighbor = False
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() in [7, 8, 16, 15, 9, 17, 35, 53]:
                        has_polar_neighbor = True
                        break
                if not has_polar_neighbor:
                    hydrophobic_count += 1
        
        descriptors['Hydrophobe'] = hydrophobic_count
        
        # Count aromatic rings
        descriptors['Aromatic'] = Descriptors.NumAromaticRings(mol)
        
        return descriptors
    
    def calculate_pharmacophore_features(self, mol):
        """Calculate pharmacophore features with fallback to basic method"""
        if self.feature_factory is not None:
            try:
                feats = self.feature_factory.GetFeaturesForMol(mol)
                
                # Count different feature types
                feature_counts = {
                    'Donor': 0, 'Acceptor': 0, 'NegIonizable': 0, 
                    'PosIonizable': 0, 'Hydrophobe': 0, 'Aromatic': 0
                }
                
                for feat in feats:
                    feat_type = feat.GetFamily()
                    if feat_type in feature_counts:
                        feature_counts[feat_type] += 1
                
                return feature_counts
            except Exception as e:
                print(f"Warning: Could not calculate advanced pharmacophore features: {e}")
        
        # Fall back to basic method
        return self.calculate_basic_pharmacophore_features(mol)
    
    def calculate_surface_descriptors(self, mol):
        """Calculate surface-related descriptors"""
        descriptors = {}
        
        # Calculate various VSA descriptors
        try:
            vsa_mr = MolSurf.SMR_VSA1(mol)  # VSA using MR
            vsa_logp = MolSurf.SlogP_VSA1(mol)  # VSA using LogP
            
            descriptors['vsa_mr'] = vsa_mr
            descriptors['vsa_logp'] = vsa_logp
        except:
            descriptors.update({'vsa_mr': 0, 'vsa_logp': 0})
            
        # Calculate PEOE_VSA descriptors
        try:
            peoe_vsa = rdMolDescriptors.PEOE_VSA1(mol)
            descriptors['peoe_vsa'] = peoe_vsa
        except:
            descriptors['peoe_vsa'] = 0
            
        return descriptors
    
    def calculate_conformational_descriptors(self, mol):
        """Calculate conformational descriptors"""
        descriptors = {}
        
        # Calculate number of rotatable bonds (flexibility)
        descriptors['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        
        # Calculate fraction of sp3 carbons (complexity)
        try:
            sp3_carbons = 0
            total_carbons = 0
            
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # Carbon
                    total_carbons += 1
                    if atom.GetHybridization() == Chem.HybridizationType.SP3:
                        sp3_carbons += 1
            
            descriptors['fraction_sp3'] = sp3_carbons / total_carbons if total_carbons > 0 else 0
        except:
            descriptors['fraction_sp3'] = 0
            
        return descriptors
    
    def calculate_deprotonation_site_descriptors(self, mol, deprotonated_atom_idx=None):
        """Calculate descriptors specific to the deprotonation site"""
        descriptors = {}
        
        if deprotonated_atom_idx is not None:
            try:
                # Get the deprotonated atom
                atom = mol.GetAtomWithIdx(deprotonated_atom_idx)
                
                # Get its neighbors
                neighbors = atom.GetNeighbors()
                
                # Calculate descriptors for the deprotonation site
                if atom.HasProp('_GasteigerCharge'):
                    descriptors['deprotonated_atom_charge'] = atom.GetDoubleProp('_GasteigerCharge')
                else:
                    descriptors['deprotonated_atom_charge'] = 0
                    
                descriptors['deprotonated_atom_degree'] = atom.GetDegree()
                
                # Count aromatic neighbors
                aromatic_neighbors = sum(1 for n in neighbors if n.GetIsAromatic())
                descriptors['aromatic_neighbors'] = aromatic_neighbors
                
                # Calculate local environment complexity
                local_atoms = set([deprotonated_atom_idx])
                for n in neighbors:
                    local_atoms.add(n.GetIdx())
                    for n2 in n.GetNeighbors():
                        local_atoms.add(n2.GetIdx())
                
                descriptors['local_env_size'] = len(local_atoms)
                
            except Exception as e:
                print(f"Warning: Could not calculate deprotonation site descriptors: {e}")
                descriptors.update({
                    'deprotonated_atom_charge': 0,
                    'deprotonated_atom_degree': 0,
                    'aromatic_neighbors': 0,
                    'local_env_size': 0
                })
                
        return descriptors
    
    def calculate_all_descriptors(self, mol, deprotonated_atom_idx=None):
        """Calculate all advanced descriptors for a molecule"""
        descriptors = {}
        
        # Basic descriptors
        descriptors['mol_weight'] = Descriptors.MolWt(mol)
        descriptors['logp'] = Descriptors.MolLogP(mol)
        descriptors['tpsa'] = Descriptors.TPSA(mol)
        descriptors['hbd'] = Lipinski.NumHDonors(mol)
        descriptors['hba'] = Lipinski.NumHAcceptors(mol)
        descriptors['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        
        # Advanced descriptors
        descriptors.update(self.calculate_3d_descriptors(mol))
        descriptors.update(self.calculate_electronic_descriptors(mol))
        descriptors.update(self.calculate_pharmacophore_features(mol))
        descriptors.update(self.calculate_surface_descriptors(mol))
        descriptors.update(self.calculate_conformational_descriptors(mol))
        
        # Deprotonation site-specific descriptors
        if deprotonated_atom_idx is not None:
            descriptors.update(self.calculate_deprotonation_site_descriptors(mol, deprotonated_atom_idx))
        
        return descriptors

# Function to identify deprotonation sites in anions
def identify_deprotonation_sites(mol):
    """Identify which atom was deprotonated in an anion"""
    # Look for oxygen atoms with negative formal charge
    deprotonated_sites = []
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == -1:  # Oxygen with negative charge
            deprotonated_sites.append(atom.GetIdx())
    
    return deprotonated_sites

# Function to calculate descriptors for all structures
def calculate_descriptors_for_all_structures():
    """Calculate advanced descriptors for all epicatechin structures"""
    calculator = AdvancedMolecularDescriptorCalculator()
    results = []
    
    # Neutral molecule
    try:
        mol = Chem.MolFromMolFile("epicatechin_neutral.mol")
        if mol:
            descriptors = calculator.calculate_all_descriptors(mol)
            results.append({
                'Molecule': 'Epicatechin_neutral',
                'Type': 'Neutral',
                'Deprotonation_Site': 'N/A',
                **descriptors
            })
    except Exception as e:
        print(f"Error processing neutral molecule: {e}")
    
    # Anions
    for i in range(1, 5):
        try:
            mol_file = f"epicatechin_mono_anion_{i}.mol"
            if os.path.exists(mol_file):
                mol = Chem.MolFromMolFile(mol_file)
                if mol:
                    # Identify deprotonation site
                    deprotonation_sites = identify_deprotonation_sites(mol)
                    deprotonation_site = deprotonation_sites[0] if deprotonation_sites else None
                    
                    # Calculate descriptors
                    descriptors = calculator.calculate_all_descriptors(mol, deprotonation_site)
                    
                    results.append({
                        'Molecule': f'Epicatechin_anion_{i}',
                        'Type': 'Anion',
                        'Deprotonation_Site': deprotonation_site,
                        **descriptors
                    })
            else:
                print(f"File not found: {mol_file}")
        except Exception as e:
            print(f"Error processing anion {i}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv("advanced_molecular_descriptors.csv", index=False)
    
    return df

# Function to analyze differences between anions
def analyze_anion_differences(descriptor_df):
    """Analyze differences between the various anionic forms"""
    # Filter to only anions
    anions_df = descriptor_df[descriptor_df['Type'] == 'Anion']
    
    if len(anions_df) < 2:
        print("Not enough anions for comparison")
        return None
    
    # Calculate mean and standard deviation for each descriptor
    descriptor_cols = [col for col in anions_df.columns if col not in ['Molecule', 'Type', 'Deprotonation_Site']]
    
    analysis_results = {}
    
    for col in descriptor_cols:
        values = anions_df[col].values
        if len(values) > 1 and not np.isnan(values).all():
            analysis_results[col] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'range': np.nanmax(values) - np.nanmin(values),
                'cv': np.nanstd(values) / np.nanmean(values) if np.nanmean(values) != 0 else np.nan
            }
    
    # Sort by coefficient of variation (highest variation first)
    sorted_results = sorted(analysis_results.items(), key=lambda x: abs(x[1]['cv'] if not np.isnan(x[1]['cv']) else 0), reverse=True)
    
    # Print top descriptors with highest variation
    print("Descriptors with highest variation between anions:")
    for i, (descriptor, stats) in enumerate(sorted_results[:10]):
        cv_display = f"{stats['cv']:.3f}" if not np.isnan(stats['cv']) else "N/A"
        print(f"{i+1}. {descriptor}: CV={cv_display}, Range={stats['range']:.3f}")
    
    # Save analysis results
    analysis_df = pd.DataFrame.from_dict(analysis_results, orient='index')
    analysis_df.to_csv("anion_descriptor_analysis.csv")
    
    return analysis_df

# Function to create a solubility prediction model using the advanced descriptors
def create_solubility_model(descriptor_df, experimental_data=None):
    """Create a solubility prediction model using the advanced descriptors"""
    # This is a placeholder for a machine learning model
    # In a real implementation, you would use experimental data to train a model
    
    print("Advanced descriptors calculated. In a real implementation, you would:")
    print("1. Collect experimental solubility data for similar compounds")
    print("2. Train a machine learning model using these descriptors as features")
    print("3. Validate the model on a test set")
    print("4. Use the model to predict solubility for your compounds")
    
    # For demonstration, we'll create a simple linear model
    # Note: This is a placeholder and not a real predictive model
    
    # Select a subset of descriptors that might be relevant for solubility
    solubility_descriptors = [
        'logp', 'tpsa', 'hbd', 'hba', 'charge_mean', 'max_neg_charge',
        'oxygen_charge_mean', 'Donor', 'Acceptor', 'vsa_mr'
    ]
    
    # Create a simple scoring function (not a real predictive model)
    def solubility_score(descriptors):
        score = (
            -0.5 * descriptors.get('logp', 0) +
            0.1 * descriptors.get('tpsa', 0) +
            -0.3 * descriptors.get('hbd', 0) +
            0.2 * descriptors.get('hba', 0) +
            2.0 * descriptors.get('max_neg_charge', 0) +
            1.5 * descriptors.get('oxygen_charge_mean', 0) +
            0.2 * descriptors.get('Donor', 0) +
            0.2 * descriptors.get('Acceptor', 0) +
            -0.01 * descriptors.get('vsa_mr', 0)
        )
        return score
    
    # Apply the scoring function
    predictions = []
    for _, row in descriptor_df.iterrows():
        score = solubility_score(row.to_dict())
        # Convert to logS scale (this is arbitrary)
        logS = -2 + score / 5
        predictions.append(logS)
    
    descriptor_df['predicted_logS'] = predictions
    
    # Save predictions
    descriptor_df.to_csv("descriptors_with_solubility_predictions.csv", index=False)
    
    return descriptor_df

# Main execution
def main():
    # Calculate advanced descriptors for all structures
    print("Calculating advanced molecular descriptors...")
    descriptor_df = calculate_descriptors_for_all_structures()
    
    if descriptor_df is not None and not descriptor_df.empty:
        # Analyze differences between anions
        print("\nAnalyzing differences between anions...")
        analysis_df = analyze_anion_differences(descriptor_df)
        
        # Create solubility predictions (placeholder model)
        print("\nCreating solubility predictions...")
        result_df = create_solubility_model(descriptor_df)
        
        # Display results
        print("\nDescriptor calculation complete. Results saved to:")
        print("- advanced_molecular_descriptors.csv")
        print("- anion_descriptor_analysis.csv")
        print("- descriptors_with_solubility_predictions.csv")
        
        # Show a sample of the results
        print("\nSample of results:")
        sample_cols = ['Molecule', 'Type', 'logp', 'tpsa', 'charge_mean', 'max_neg_charge', 'predicted_logS']
        print(result_df[sample_cols].head().to_string(index=False))
    else:
        print("No descriptors were calculated. Please check if the input files exist.")

if __name__ == "__main__":
    main()

"""
Key Features of This Implementation:
Comprehensive Descriptor Set: Calculates a wide range of molecular descriptors including:

3D descriptors (radius of gyration, principal moments of inertia)
Electronic descriptors (charge distributions, oxygen charge statistics)
Pharmacophore features (donors, acceptors, hydrophobic regions)
Surface descriptors (VSA, PEOE_VSA)
Conformational descriptors (flexibility, complexity)
Deprotonation site-specific descriptors
Anion Differentiation: Specifically designed to capture differences between various anionic forms by:
Identifying the deprotonation site
Calculating local environment descriptors around the deprotonation site
Analyzing charge distribution patterns
Statistical Analysis: Includes functionality to analyze which descriptors vary most between different anions, helping identify the most important factors for differentiation.
Flexible Framework: Provides a foundation for building more sophisticated solubility prediction models that can incorporate experimental data.

How to Use:
Run the script after generating your molecular structures:

bash
python advanced_descriptors.py
The script will:
Calculate advanced descriptors for all structures
Analyze differences between anions
Create solubility predictions using a placeholder model
Save results to CSV files
For real applications, you would:
Collect experimental solubility data for similar compounds
Use the calculated descriptors as features in a machine learning model
Train and validate the model on your experimental data
Use the trained model to make predictions for your compounds

Expected Benefits:
Better Differentiation: The advanced descriptors should better capture the subtle differences between various anionic forms of epicatechin.
Improved Predictions: A model trained on these descriptors should provide more accurate solubility predictions.
Mechanistic Insights: The analysis of which descriptors vary most between anions can provide insights into the molecular factors that influence solubility.
This implementation provides a comprehensive framework for calculating advanced molecular descriptors and using them to differentiate between various anionic forms of epicatechin, which should lead to more accurate and nuanced solubility predictions.

"""
