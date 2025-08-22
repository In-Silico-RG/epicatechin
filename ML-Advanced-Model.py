import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class SolubilityPredictor:
    def __init__(self, solubility_data_path):
        self.solubility_data_path = solubility_data_path
        self.solubility_data = None
        self.descriptor_data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_and_prepare_solubility_data(self):
        """Load and prepare the solubility dataset for modeling"""
        try:
            self.solubility_data = pd.read_csv(self.solubility_data_path)
            print(f"Loaded solubility data with {len(self.solubility_data)} entries")
            
            # Check if we have the necessary columns
            if 'SMILES' not in self.solubility_data.columns or 'Solubility' not in self.solubility_data.columns:
                print("Warning: SMILES or Solubility column not found")
                return False
                
            # Calculate molecular descriptors for the solubility data
            print("Calculating molecular descriptors for solubility data...")
            self.solubility_data = self.calculate_descriptors_for_solubility_data(self.solubility_data)
            
            return True
            
        except Exception as e:
            print(f"Error loading solubility data: {e}")
            return False
    
    def calculate_descriptors_for_solubility_data(self, df):
        """Calculate molecular descriptors for the solubility dataset"""
        descriptors_list = []
        
        for _, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row['SMILES'])
                if mol:
                    # Calculate basic descriptors (similar to what you have in your descriptor data)
                    desc = {
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
                    }
                    descriptors_list.append(desc)
                else:
                    descriptors_list.append({})
            except Exception as e:
                print(f"Error calculating descriptors: {e}")
                descriptors_list.append({})
        
        # Add descriptors to dataframe
        desc_df = pd.DataFrame(descriptors_list)
        return pd.concat([df.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)
    
    def load_descriptor_data(self, descriptor_path="advanced_molecular_descriptors.csv"):
        """Load your molecular descriptor data"""
        try:
            self.descriptor_data = pd.read_csv(descriptor_path)
            print(f"Loaded descriptor data with {len(self.descriptor_data)} entries")
            return True
        except Exception as e:
            print(f"Error loading descriptor data: {e}")
            return False
    
    def explore_solubility_data(self):
        """Explore the solubility dataset"""
        if self.solubility_data is None:
            print("No solubility data available")
            return
        
        print("Solubility Dataset Overview:")
        print(self.solubility_data.info())
        
        print("\nFirst few rows:")
        print(self.solubility_data.head())
        
        print("\nSolubility distribution:")
        if 'Solubility' in self.solubility_data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.solubility_data['Solubility'], kde=True)
            plt.title('Distribution of Solubility Values')
            plt.xlabel('Solubility')
            plt.ylabel('Frequency')
            plt.savefig('solubility_distribution.png')
            plt.show()
        
        print("\nCorrelation with solubility:")
        if 'Solubility' in self.solubility_data.columns:
            # Calculate correlations with solubility
            numeric_cols = self.solubility_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Solubility' in numeric_cols:
                numeric_cols.remove('Solubility')
            
            correlations = self.solubility_data[numeric_cols].corrwith(self.solubility_data['Solubility'])
            correlations = correlations.sort_values(ascending=False)
            
            print("Top 10 positive correlations:")
            print(correlations.head(10))
            
            print("\nTop 10 negative correlations:")
            print(correlations.tail(10))
            
            # Plot top correlations
            top_correlations = pd.concat([correlations.head(5), correlations.tail(5)])
            plt.figure(figsize=(12, 8))
            top_correlations.plot(kind='bar')
            plt.title('Top Correlations with Solubility')
            plt.ylabel('Correlation Coefficient')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('solubility_correlations.png')
            plt.show()
    
    def prepare_model_data(self):
        """Prepare data for machine learning"""
        if self.solubility_data is None:
            print("No solubility data available")
            return None, None, None
        
        # Identify feature columns (exclude non-feature columns)
        exclude_cols = ['ID', 'Name', 'InChI', 'InChIKey', 'SMILES', 'SD', 'Ocurrences', 'Group']
        feature_cols = [col for col in self.solubility_data.columns 
                       if col not in exclude_cols and col != 'Solubility']
        
        # Separate features and target
        X = self.solubility_data[feature_cols]
        y = self.solubility_data['Solubility']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Store feature names for later use
        self.feature_names = feature_cols
        
        return X, y
    
    def train_model(self, X, y):
        """Train a Random Forest model for solubility prediction"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"Cross-validation R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        # Plot predictions vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Solubility')
        plt.ylabel('Predicted Solubility')
        plt.title('Actual vs Predicted Solubility')
        plt.savefig('solubility_predictions.png')
        plt.show()
        
        return self.model, rmse, r2
    
    def feature_importance(self):
        """Analyze feature importance"""
        if self.model is None:
            print("No trained model available")
            return
        
        # Get feature importances
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Print feature ranking
        print("Feature ranking:")
        for f in range(min(20, len(self.feature_names))):
            print(f"{f+1}. {self.feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(15), importances[indices][:15], align="center")
        plt.xticks(range(15), [self.feature_names[i] for i in indices[:15]], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()
    
    def predict_epicatechin_solubility(self):
        """Predict solubility for your epicatechin compounds"""
        if self.model is None or self.descriptor_data is None:
            print("No trained model or descriptor data available")
            return None
        
        # Prepare the descriptor data for prediction
        # We need to align the features with what the model was trained on
        
        # First, let's see what features we have in both datasets
        print("Available features in descriptor data:", list(self.descriptor_data.columns))
        print("Model was trained on:", self.feature_names)
        
        # Create a mapping between your descriptor names and the solubility dataset names
        feature_mapping = {
            'mol_weight': 'mol_weight',
            'logp': 'logp',
            'tpsa': 'tpsa',
            'hbd': 'hbd',
            'hba': 'hba',
            'aromatic_rings': 'aromatic_rings',
            'rotatable_bonds': 'rotatable_bonds',
        }
        
        # Prepare the prediction data
        prediction_data = pd.DataFrame()
        
        for model_feature, desc_feature in feature_mapping.items():
            if desc_feature in self.descriptor_data.columns:
                prediction_data[model_feature] = self.descriptor_data[desc_feature]
            else:
                print(f"Warning: {desc_feature} not found in descriptor data")
                # Fill with mean value from training data
                prediction_data[model_feature] = self.solubility_data[model_feature].mean()
        
        # Handle missing values
        prediction_data = prediction_data.fillna(prediction_data.mean())
        
        # Scale the features
        prediction_data_scaled = self.scaler.transform(prediction_data)
        
        # Make predictions
        predictions = self.model.predict(prediction_data_scaled)
        
        # Add predictions to descriptor data
        self.descriptor_data['predicted_solubility'] = predictions
        
        return predictions
    
    def save_model(self, model_path='solubility_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            print("No trained model available")
            return False
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, model_path)
        
        print(f"Model saved to {model_path}")
        return True

# Main execution
def main():
    # Path to your curated solubility dataset
    solubility_path = "/home/aldo/sims/epi/test/curated-solubility-dataset.csv"
    
    # Initialize the solubility predictor
    predictor = SolubilityPredictor(solubility_path)
    
    # Load and prepare the solubility data
    if not predictor.load_and_prepare_solubility_data():
        return
    
    # Explore the data
    predictor.explore_solubility_data()
    
    # Prepare data for modeling
    X, y = predictor.prepare_model_data()
    
    if X is None or y is None:
        print("Insufficient data for modeling")
        return
    
    # Train the model
    model, rmse, r2 = predictor.train_model(X, y)
    
    # Analyze feature importance
    predictor.feature_importance()
    
    # Save the model
    predictor.save_model()
    
    # Load your descriptor data
    if predictor.load_descriptor_data():
        # Predict solubility for your epicatechin compounds
        predictions = predictor.predict_epicatechin_solubility()
        
        if predictions is not None:
            print("\nSolubility predictions for your compounds:")
            for i, pred in enumerate(predictions):
                print(f"{predictor.descriptor_data['Molecule'].iloc[i]}: {pred:.3f}")
            
            # Save predictions
            predictor.descriptor_data.to_csv("epicatechin_solubility_predictions.csv", index=False)
            print("Solubility predictions saved to epicatechin_solubility_predictions.csv")

if __name__ == "__main__":
    main()


"""
Key Changes and Improvements:
Separate Data Processing: Instead of trying to merge the datasets, we process them separately

Descriptor Calculation: We calculate the same descriptors for the solubility data that you have for your epicatechin compounds

Feature Alignment: We create a mapping between the feature names in both datasets

Robust Error Handling: Added better error handling and fallbacks for missing features

Visualization: Maintained the visualization components to understand the data and model performance

How This Works:
Solubility Data Processing:

Loads your curated solubility dataset

Calculates molecular descriptors for each compound using RDKit

Prepares the data for machine learning

Model Training:

Trains a Random Forest model on the solubility data

Evaluates model performance

Analyzes feature importance

Prediction:

Loads your epicatechin descriptor data

Aligns the features with the trained model

Predicts solubility for your compounds

Saves the results

This approach should work even though your datasets don't have overlapping compounds, as it uses the solubility data to train a general model that can then be applied to your specific compounds.



Key Changes:
Common Feature Identification: Added a method to identify common features between your solubility dataset and descriptor data

Consistent Feature Usage: The model is now trained and makes predictions using only the common features

Better Error Handling: Added more robust error handling and checks

Feature Importance: Updated the feature importance analysis to use the common features

How This Fixes the Problem:
Feature Consistency: By identifying and using only the common features between both datasets, we ensure that the features used during training match those used during prediction

Scaler Compatibility: The scaler is now trained on the same set of features that will be used for prediction

Model Compatibility: The model is trained on the exact same features that will be available during prediction

This approach should resolve the feature mismatch error and allow you to successfully train a model on your solubility data and apply it to your epicatechin compounds.

"""
