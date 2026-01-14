"""
Data processing utilities for clinical readmission risk analysis.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic clinical data for readmission risk analysis.
    
    Args:
        n_samples: Number of sample patients to generate
        
    Returns:
        DataFrame with clinical features and readmission status
    """
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 95, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples, p=[0.5, 0.3, 0.2]),
        'num_medications': np.random.randint(0, 20, n_samples),
        'num_procedures': np.random.randint(0, 10, n_samples),
        'time_in_hospital': np.random.randint(1, 14, n_samples),
        'num_lab_procedures': np.random.randint(10, 100, n_samples),
        'num_diagnoses': np.random.randint(1, 16, n_samples),
        'previous_visits': np.random.randint(0, 10, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'heart_disease': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate readmission target based on features with some logic
    readmission_prob = (
        (df['age'] > 65) * 0.15 +
        (df['num_medications'] > 10) * 0.15 +
        (df['time_in_hospital'] > 7) * 0.1 +
        (df['previous_visits'] > 3) * 0.2 +
        (df['diabetes'] == 1) * 0.1 +
        (df['hypertension'] == 1) * 0.1 +
        (df['heart_disease'] == 1) * 0.1 +
        (df['admission_type'] == 'Emergency') * 0.1 +
        np.random.random(n_samples) * 0.2
    )
    
    df['readmitted'] = (readmission_prob > 0.5).astype(int)
    
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess clinical data for machine learning.
    
    Args:
        df: Raw clinical data DataFrame
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Separate features and target
    if 'readmitted' in df.columns:
        X = df.drop(['readmitted', 'patient_id'], axis=1, errors='ignore')
        y = df['readmitted']
    else:
        X = df.drop(['patient_id'], axis=1, errors='ignore')
        y = None
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X, y


def calculate_risk_score(prediction_proba: float) -> str:
    """
    Convert prediction probability to risk category.
    
    Args:
        prediction_proba: Probability of readmission (0-1)
        
    Returns:
        Risk category string
    """
    if prediction_proba < 0.3:
        return "Low Risk"
    elif prediction_proba < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


def get_feature_importance_interpretation(feature_name: str) -> str:
    """
    Get human-readable interpretation of feature importance.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Interpretation string
    """
    interpretations = {
        'age': 'Patient age',
        'num_medications': 'Number of medications',
        'num_procedures': 'Number of procedures',
        'time_in_hospital': 'Length of hospital stay',
        'num_lab_procedures': 'Number of lab procedures',
        'num_diagnoses': 'Number of diagnoses',
        'previous_visits': 'Previous hospital visits',
        'diabetes': 'Diabetes diagnosis',
        'hypertension': 'Hypertension diagnosis',
        'heart_disease': 'Heart disease diagnosis',
        'gender_M': 'Male gender',
        'admission_type_Urgent': 'Urgent admission',
        'admission_type_Emergency': 'Emergency admission',
    }
    
    return interpretations.get(feature_name, feature_name.replace('_', ' ').title())
