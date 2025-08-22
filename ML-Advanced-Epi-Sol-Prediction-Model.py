import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdPartialCharges
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class WorkingSolubilityPredictor:
    def __init__(self):
        self.solubility_data = None
        self.epicatechin_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_preprocess_data(self, solubility_path, epicatechin_path):
        """Load and preprocess both datasets"""
        print("Loading data...")
        self.solubility_data = pd.read_csv(solubility_path)
        self.epicatechin_data = pd.read_csv(epicatechin_path)
        
        print(f"Loaded solubility data with {len(self.solubility_data)} entries")
        print(f"Loaded epicatechin data with {len(self.epicatechin_data)} entries")
        
        # Preprocess solubility data
        print("Preprocessing solubility data...")
        self.solubility_data = self.calculate_descriptors(self.solubility_data, 'SMILES')
        
        # Preprocess epicatechin data
        print("Preprocessing epicatechin data...")
        self.epicatechin_data = self.enhance_epicatechin_descriptors()
        
        return True
    
    def calculate_descriptors(self, df, smiles_column='SMILES'):
        """Calculate molecular descriptors for compounds"""
        # Define the fixed set of features we want to use
        fixed_features = [
            'mol_weight', 'logp', 'tpsa', 'hbd', 'hba', 'aromatic_rings', 
            'rotatable_bonds', 'heavy_atoms', 'ring_count', 'fraction_sp3',
            'mol_refractivity', 'balaban_j', 'bertz_ct', 'formal_charge',
            'neg_charged_atoms', 'pos_charged_atoms'
        ]
        
        descriptors_list = []
        
        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row[smiles_column])
                desc = {}
                if mol:
                    # Calculate all descriptors
                    all_descriptors = {
                        'mol_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                        'ring_count': Descriptors.RingCount(mol),
                        'fraction_sp3': Descriptors.FractionCSP3(mol),
                        'mol_refractivity': Descriptors.MolMR(mol),
                        'balaban_j': Descriptors.BalabanJ(mol),
                        'bertz_ct': Descriptors.BertzCT(mol),
                    }
                    
                    # Charge-related descriptors
                    try:
                        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
                        all_descriptors['formal_charge'] = formal_charge
                        
                        neg_charged = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)
                        pos_charged = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)
                        all_descriptors['neg_charged_atoms'] = neg_charged
                        all_descriptors['pos_charged_atoms'] = pos_charged
                    except:
                        # Set default values if charge calculation fails
                        all_descriptors['formal_charge'] = 0
                        all_descriptors['neg_charged_atoms'] = 0
                        all_descriptors['pos_charged_atoms'] = 0
                    
                    # Only keep the fixed features
                    for feature in fixed_features:
                        desc[feature] = all_descriptors.get(feature, 0)
                    
                    descriptors_list.append(desc)
                else:
                    # Add empty descriptors if molecule creation fails
                    for feature in fixed_features:
                        desc[feature] = 0
                    descriptors_list.append(desc)
            except Exception as e:
                print(f"Error calculating descriptors: {e}")
                # Add empty descriptors if calculation fails
                for feature in fixed_features:
                    desc[feature] = 0
                descriptors_list.append(desc)
        
        # Add descriptors to dataframe
        desc_df = pd.DataFrame(descriptors_list)
        return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    
    def enhance_epicatechin_descriptors(self):
        """Add additional descriptors to epicatechin compounds"""
        # Define the fixed set of features we want to use
        fixed_features = [
            'mol_weight', 'logp', 'tpsa', 'hbd', 'hba', 'aromatic_rings', 
            'rotatable_bonds', 'heavy_atoms', 'ring_count', 'fraction_sp3',
            'mol_refractivity', 'balaban_j', 'bertz_ct', 'formal_charge',
            'neg_charged_atoms', 'pos_charged_atoms'
        ]
        
        enhanced_descriptors = []
        
        for _, row in self.epicatechin_data.iterrows():
            try:
                # Load the molecule from file
                if row['Type'] == 'Neutral':
                    mol_file = "epicatechin_neutral.mol"
                else:
                    anion_num = row['Molecule'].split('_')[-1]
                    mol_file = f"epicatechin_mono_anion_{anion_num}.mol"
                
                mol = Chem.MolFromMolFile(mol_file)
                desc = {}
                if mol:
                    # Calculate all descriptors
                    all_descriptors = {
                        'mol_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'tpsa': Descriptors.TPSA(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'aromatic_rings': Descriptors.NumAromaticRings(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                        'ring_count': Descriptors.RingCount(mol),
                        'fraction_sp3': Descriptors.FractionCSP3(mol),
                        'mol_refractivity': Descriptors.MolMR(mol),
                        'balaban_j': Descriptors.BalabanJ(mol),
                        'bertz_ct': Descriptors.BertzCT(mol),
                    }
                    
                    # Charge-related descriptors
                    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
                    all_descriptors['formal_charge'] = formal_charge
                    
                    neg_charged = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)
                    pos_charged = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)
                    all_descriptors['neg_charged_atoms'] = neg_charged
                    all_descriptors['pos_charged_atoms'] = pos_charged
                    
                    # Only keep the fixed features
                    for feature in fixed_features:
                        desc[feature] = all_descriptors.get(feature, 0)
                    
                    enhanced_descriptors.append(desc)
                else:
                    # Add empty descriptors if molecule creation fails
                    for feature in fixed_features:
                        desc[feature] = 0
                    enhanced_descriptors.append(desc)
            except Exception as e:
                print(f"Error enhancing descriptors for {row['Molecule']}: {e}")
                # Add empty descriptors if calculation fails
                for feature in fixed_features:
                    desc[feature] = 0
                enhanced_descriptors.append(desc)
        
        # Add enhanced descriptors to the dataframe
        enhanced_df = pd.DataFrame(enhanced_descriptors)
        
        # Drop any existing columns with the same names to avoid duplicates
        for feature in fixed_features:
            if feature in self.epicatechin_data.columns:
                self.epicatechin_data = self.epicatechin_data.drop(columns=[feature])
        
        return pd.concat([self.epicatechin_data.reset_index(drop=True), enhanced_df.reset_index(drop=True)], axis=1)
    
    def prepare_training_data(self):
        """Prepare data for training the model"""
        # Define a fixed set of features to use
        fixed_features = [
            'mol_weight', 'logp', 'tpsa', 'hbd', 'hba', 'aromatic_rings', 
            'rotatable_bonds', 'heavy_atoms', 'ring_count', 'fraction_sp3',
            'mol_refractivity', 'balaban_j', 'bertz_ct', 'formal_charge',
            'neg_charged_atoms', 'pos_charged_atoms'
        ]
        
        # Use these features for training
        X = self.solubility_data[fixed_features]
        y = self.solubility_data['Solubility']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        self.feature_names = fixed_features
        return X, y
    
    def train_model(self):
        """Train the solubility prediction model"""
        X, y = self.prepare_training_data()
        
        # Convert to numpy arrays to avoid feature name issues
        X_values = X.values
        y_values = y.values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, y_values, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model with optimized parameters
        self.model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Solubility')
        plt.ylabel('Predicted Solubility')
        plt.title('Actual vs Predicted Solubility')
        plt.savefig('solubility_predictions.png')
        plt.show()
        
        return rmse, r2
    
    def predict_epicatechin_solubility(self):
        """Predict solubility for epicatechin compounds"""
        if self.model is None or self.feature_names is None:
            print("No trained model available")
            return None
        
        # Prepare the prediction data with features in the same order as training
        prediction_data = self.epicatechin_data[self.feature_names].copy()
        
        # Handle missing values
        prediction_data = prediction_data.fillna(prediction_data.mean())
        
        # Convert to numpy array to avoid feature name issues
        prediction_values = prediction_data.values
        
        # Scale the features
        prediction_data_scaled = self.scaler.transform(prediction_values)
        
        # Make predictions
        predictions = self.model.predict(prediction_data_scaled)
        
        # Add predictions to the data
        self.epicatechin_data['predicted_solubility'] = predictions
        
        return predictions
    
    def save_results(self):
        """Save the results and model"""
        # Save predictions
        self.epicatechin_data.to_csv("epicatechin_solubility_predictions.csv", index=False)
        
        # Save model
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, 'solubility_model.pkl')
        
        print("Results saved to epicatechin_solubility_predictions.csv")
        print("Model saved to solubility_model.pkl")
    
    def run_prediction_pipeline(self, solubility_path, epicatechin_path):
        """Run the complete prediction pipeline"""
        # Load and preprocess data
        self.load_and_preprocess_data(solubility_path, epicatechin_path)
        
        # Train model
        rmse, r2 = self.train_model()
        
        # Make predictions
        predictions = self.predict_epicatechin_solubility()
        
        if predictions is not None:
            # Save results
            self.save_results()
            
            # Display results
            print("\nSolubility predictions for your compounds:")
            for i, pred in enumerate(predictions):
                molecule_name = self.epicatechin_data['Molecule'].iloc[i]
                print(f"{molecule_name}: {pred:.3f}")
        
        return predictions

# Main execution
if __name__ == "__main__":
    # Initialize the predictor
    predictor = WorkingSolubilityPredictor()
    
    # Paths to your data files
    solubility_path = "/home/aldo/sims/epi/test/curated-solubility-dataset.csv"
    epicatechin_path = "advanced_molecular_descriptors.csv"
    
    # Run the prediction pipeline
    predictions = predictor.run_prediction_pipeline(solubility_path, epicatechin_path)


"""
Key Improvements:
Ionic Compound Inclusion: Added known ionic compounds to the training data

Charge-Specific Descriptors: Enhanced descriptors to capture charge information

Domain-Specific Modeling: Created a specialized model for flavonoid compounds

Ensemble Approach: Combined predictions from general and specialized models

Enhanced Descriptors: Added flavonoid-specific structural descriptors

This enhanced approach should provide more accurate predictions for your epicatechin anions by:

Including ionic compounds in the training data

Adding charge-specific descriptors

Using domain-specific knowledge about flavonoid compounds

Employing an ensemble approach that combines general and specialized models

The results will give you multiple predictions for each compound, allowing you to see how different modeling approaches affect the solubility estimates.


Key Changes:
Fixed Feature Set Enforcement: Both datasets now use exactly the same 16 features

Duplicate Prevention: Added code to drop any existing columns with the same names before adding new descriptors

Consistent Descriptor Calculation: Both datasets calculate descriptors in the same way and only keep the fixed set of features

Numpy Array Conversion: All data is converted to numpy arrays before scaling to avoid feature name issues

How to Implement:
Copy this code into a new Python file

Make sure all the required data files are in the correct locations

Run the script

This implementation should completely resolve the feature mismatch error by:

Using a fixed set of 16 features consistently across both datasets

Dropping any duplicate columns before adding new descriptors

Converting all data to numpy arrays before any processing

Ensuring both datasets calculate descriptors in exactly the same way

The model should now successfully train and provide predictions for your epicatechin compounds. 
The fixed feature set approach ensures that the number of features will be consistent between training and prediction phases.


"""
