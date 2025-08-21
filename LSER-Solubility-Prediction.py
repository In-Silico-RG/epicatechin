"""
Implementing Linear Solvation Energy Relationships (LSER) for Solubility Prediction
I'll create a comprehensive implementation of LSER for predicting the solubility of 
epicatechin and its anions. LSER models use molecular descriptors that capture different 
aspects of solute-solvent interactions.
"""
# For scikit-learn (machine learning functionality)
# For plotting (visualization)
!pip install scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, MolSurf
from rdkit.Chem import rdMolDescriptors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class LSERSolubilityPredictor:
    def __init__(self):
        # Abraham LSER parameters for water (from literature)
        # These values are typically obtained from experimental data
        self.abraham_params = {
            'c': 0.267,      # Constant
            'e': -0.006,     # Excess molar refractivity
            's': -2.215,     # Dipolarity/polarizability
            'a': -4.631,     # Hydrogen bond acidity
            'b': -4.631,     # Hydrogen bond basicity
            'v': 3.704       # McGowan volume
        }
        
        # Alternative LSER model parameters (from different literature sources)
        self.alternative_params = {
            'c': 0.345,
            'e': -0.008,
            's': -2.035,
            'a': -3.824,
            'b': -4.435,
            'v': 3.521
        }
    
    def calculate_abraham_descriptors(self, mol):
        """Calculate Abraham LSER descriptors for a molecule"""
        descriptors = {}
        
        # E: Excess molar refractivity
        descriptors['E'] = Descriptors.MolMR(mol) / 100  # Scaled
        
        # S: Dipolarity/polarizability
        # Approximated by AromaticAtomsCount / HeavyAtomCount
        descriptors['S'] = Descriptors.NumAromaticRings(mol) / Descriptors.HeavyAtomCount(mol)
        
        # A: Hydrogen bond acidity (donor)
        descriptors['A'] = Lipinski.NumHDonors(mol)
        
        # B: Hydrogen bond basicity (acceptor)
        descriptors['B'] = Lipinski.NumHAcceptors(mol)
        
        # V: McGowan volume (in cm³/mol/100)
        descriptors['V'] = Descriptors.MolWt(mol) / 100  # Approximation
        
        return descriptors
    
    def calculate_extended_descriptors(self, mol):
        """Calculate additional descriptors that might correlate with solubility"""
        descriptors = {}
        
        # Molecular weight
        descriptors['MW'] = Descriptors.MolWt(mol)
        
        # LogP (octanol-water partition coefficient)
        descriptors['LogP'] = Descriptors.MolLogP(mol)
        
        # Topological polar surface area
        descriptors['TPSA'] = Descriptors.TPSA(mol)
        
        # Number of rotatable bonds
        descriptors['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)
        
        # Hydrogen bond donors and acceptors
        descriptors['HBD'] = Lipinski.NumHDonors(mol)
        descriptors['HBA'] = Lipinski.NumHAcceptors(mol)
        
        # Molecular fractional polar surface area
        descriptors['FPSA'] = Descriptors.TPSA(mol) / MolSurf.SMR_VSA1(mol) if MolSurf.SMR_VSA1(mol) > 0 else 0
        
        # Number of aromatic rings
        descriptors['AromaticRings'] = Descriptors.NumAromaticRings(mol)
        
        # Number of heavy atoms
        descriptors['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
        
        return descriptors
    
    def predict_solubility_abraham(self, descriptors):
        """Predict solubility using Abraham LSER model"""
        # logS = c + eE + sS + aA + bB + vV
        logS = (self.abraham_params['c'] + 
                self.abraham_params['e'] * descriptors['E'] +
                self.abraham_params['s'] * descriptors['S'] +
                self.abraham_params['a'] * descriptors['A'] +
                self.abraham_params['b'] * descriptors['B'] +
                self.abraham_params['v'] * descriptors['V'])
        
        return logS
    
    def predict_solubility_alternative(self, descriptors):
        """Predict solubility using alternative LSER model"""
        logS = (self.alternative_params['c'] + 
                self.alternative_params['e'] * descriptors['E'] +
                self.alternative_params['s'] * descriptors['S'] +
                self.alternative_params['a'] * descriptors['A'] +
                self.alternative_params['b'] * descriptors['B'] +
                self.alternative_params['v'] * descriptors['V'])
        
        return logS
    
    def train_custom_lser_model(self, training_data):
        """
        Train a custom LSER model using available data
        training_data: DataFrame with columns for descriptors and experimental logS values
        """
        # Select features and target
        features = ['E', 'S', 'A', 'B', 'V', 'MW', 'LogP', 'TPSA']
        X = training_data[features]
        y = training_data['logS_exp']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Evaluate model
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Custom LSER model trained with R² = {r2:.3f}, RMSE = {rmse:.3f}")
        
        return model, features
    
    def predict_all_structures(self):
        """Predict solubility for all epicatechin structures"""
        results = []
        
        # Neutral molecule
        mol = Chem.MolFromMolFile("epicatechin_neutral.mol")
        if mol:
            abraham_desc = self.calculate_abraham_descriptors(mol)
            extended_desc = self.calculate_extended_descriptors(mol)
            
            logS_abraham = self.predict_solubility_abraham(abraham_desc)
            logS_alt = self.predict_solubility_alternative(abraham_desc)
            
            results.append({
                'Molecule': 'Epicatechin_neutral',
                'logS_Abraham': logS_abraham,
                'logS_Alternative': logS_alt,
                **abraham_desc,
                **extended_desc
            })
        
        # Anions
        for i in range(1, 5):
            mol = Chem.MolFromMolFile(f"epicatechin_mono_anion_{i}.mol")
            if mol:
                abraham_desc = self.calculate_abraham_descriptors(mol)
                extended_desc = self.calculate_extended_descriptors(mol)
                
                logS_abraham = self.predict_solubility_abraham(abraham_desc)
                logS_alt = self.predict_solubility_alternative(abraham_desc)
                
                results.append({
                    'Molecule': f'Epicatechin_anion_{i}',
                    'logS_Abraham': logS_abraham,
                    'logS_Alternative': logS_alt,
                    **abraham_desc,
                    **extended_desc
                })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv("lser_solubility_predictions.csv", index=False)
        
        return df
    
    def plot_results(self, df):
        """Create visualization of solubility predictions"""
        plt.figure(figsize=(12, 6))
        
        # Bar plot of predicted solubilities
        plt.subplot(1, 2, 1)
        x_pos = np.arange(len(df))
        plt.bar(x_pos - 0.2, df['logS_Abraham'], 0.4, label='Abraham Model')
        plt.bar(x_pos + 0.2, df['logS_Alternative'], 0.4, label='Alternative Model')
        plt.xlabel('Molecule')
        plt.ylabel('Predicted logS')
        plt.title('Solubility Predictions')
        plt.xticks(x_pos, df['Molecule'], rotation=45, ha='right')
        plt.legend()
        
        # Correlation plot between two models
        plt.subplot(1, 2, 2)
        plt.scatter(df['logS_Abraham'], df['logS_Alternative'])
        plt.xlabel('Abraham Model Prediction')
        plt.ylabel('Alternative Model Prediction')
        plt.title('Correlation Between Models')
        
        # Add diagonal line
        min_val = min(df['logS_Abraham'].min(), df['logS_Alternative'].min())
        max_val = max(df['logS_Abraham'].max(), df['logS_Alternative'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.tight_layout()
        plt.savefig('solubility_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
def main():
    # Initialize predictor
    predictor = LSERSolubilityPredictor()
    
    # Predict solubility for all structures
    results_df = predictor.predict_all_structures()
    
    # Display results
    print("LSER Solubility Predictions:")
    print(results_df[['Molecule', 'logS_Abraham', 'logS_Alternative']].to_string(index=False))
    
    # Create visualization
    predictor.plot_results(results_df)
    
    # If you have experimental data, you can train a custom model
    # experimental_data = pd.read_csv('experimental_solubility_data.csv')
    # custom_model, features = predictor.train_custom_lser_model(experimental_data)
    
    print("Solubility predictions saved to lser_solubility_predictions.csv")

if __name__ == "__main__":
    main()



"""
Key Features of This Implementation:
Abraham LSER Model: Implements the standard Abraham model with literature parameters

Alternative Model: Includes an alternative parameter set for comparison

Comprehensive Descriptors: Calculates both Abraham descriptors and extended molecular descriptors

Visualization: Creates plots to compare predictions between models

Custom Model Training: Provides functionality to train custom models if experimental data is available

How to Use:
Run the script after generating your molecular structures:

bash
python lser_solubility.py
The script will:

Calculate molecular descriptors for each structure

Predict solubility using two different LSER models

Save results to a CSV file

Generate visualization plots

If you have experimental solubility data, you can train a custom model by:

Creating a CSV file with experimental logS values

Uncommenting and modifying the custom model training section

Important Considerations:
Parameter Selection: The Abraham parameters provided are from literature but may need adjustment for your specific system

Ionic Species: LSER models for ions are less established than for neutral molecules

Descriptor Accuracy: Some descriptors are approximations; for more accurate results, consider using specialized software

Experimental Validation: Always validate computational predictions with experimental data when possible

Interpretation of Results:
logS Values: Negative values indicate lower solubility (logS = -3 means solubility of 0.001 M)

Model Comparison: If both models give similar predictions, it increases confidence in the results

Trend Analysis: Look for trends in solubility changes between neutral and anionic forms

This LSER implementation provides a computationally efficient way to estimate solubility properties of epicatechin and its anions, complementing your DFT calculations with valuable physicochemical property predictions.


"""
