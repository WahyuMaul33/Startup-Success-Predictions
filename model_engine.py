import joblib
import pandas as pd
import numpy as np

class UMKMModelWrapper:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        
        self.feature_names = [
            'age_first_funding_year', 'age_last_funding_year', 
            'age_first_milestone_year', 'age_last_milestone_year', 
            'funding_rounds', 'funding_total_usd', 'milestones', 
            'is_CA', 'is_NY', 'is_MA', 'is_TX', 'is_otherstate', 
            'is_software', 'is_web', 'is_mobile', 'is_enterprise', 
            'is_advertising', 'is_gamesvideo', 'is_ecommerce', 
            'is_biotech', 'is_consulting', 'is_othercategory', 
            'has_VC', 'has_angel', 'has_roundA', 'has_roundB', 
            'has_roundC', 'has_roundD', 'avg_participants', 
            'is_top500', 'has_RoundABCD', 'has_Investor', 
            'has_Seed', 'invalid_startup', 'age_startup_year', 
            'tier_relationships'
        ]

    def _pad_features(self, user_input: dict):
        """Ensures the input has exactly 36 columns in the correct order."""
        full_data = {feat: 0 for feat in self.feature_names}
        
        full_data.update(user_input)
        
        return pd.DataFrame([full_data])[self.feature_names]

    def predict(self, user_input: dict):
        final_input = self._pad_features(user_input)
        
        prediction = self.model.predict(final_input)[0]
        probability = self.model.predict_proba(final_input)[0][1]
        
        return {
            "prediction_status": "Growth/Success" if prediction == 1 else "At Risk",
            "survival_probability": round(float(probability), 4),
            "model_metadata": {
                "architecture": "GradientBoosting",
                "training_accuracy": 0.881
            }
        }