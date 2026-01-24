#!/usr/bin/env python3
"""
Example script demonstrating the clinical readmission risk explorer functionality.
This script shows how to use the core modules without the Streamlit UI.
"""

from data_processing import generate_sample_data, preprocess_data, calculate_risk_score
from model import ReadmissionRiskModel
import pandas as pd


def main():
    """Demonstrate the clinical readmission risk explorer."""
    
    print("=" * 70)
    print("Clinical Readmission Risk Explorer - Example Usage")
    print("=" * 70)
    
    # 1. Generate synthetic patient data
    print("\n1. Generating synthetic patient data...")
    df = generate_sample_data(n_samples=500)
    print(f"   Generated {len(df)} patient records")
    print(f"   Readmission rate: {df['readmitted'].sum() / len(df):.1%}")
    
    # 2. Preprocess the data
    print("\n2. Preprocessing data for machine learning...")
    X, y = preprocess_data(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Number of features: {X.shape[1]}")
    
    # 3. Train the model
    print("\n3. Training Random Forest model...")
    model = ReadmissionRiskModel()
    metrics = model.train(X, y)
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"   Training samples: {metrics['train_samples']}")
    print(f"   Test samples: {metrics['test_samples']}")
    
    # 4. Get feature importance
    print("\n4. Top 5 most important features:")
    importance_df = model.get_feature_importance()
    for idx, row in importance_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # 5. Make predictions for sample patients
    print("\n5. Making predictions for sample patients:")
    
    # High-risk patient
    high_risk_patient = pd.DataFrame({
        'patient_id': [1],
        'age': [75],
        'gender': ['M'],
        'admission_type': ['Emergency'],
        'num_medications': [15],
        'num_procedures': [5],
        'time_in_hospital': [10],
        'num_lab_procedures': [80],
        'num_diagnoses': [12],
        'previous_visits': [5],
        'diabetes': [1],
        'hypertension': [1],
        'heart_disease': [1]
    })
    
    # Low-risk patient
    low_risk_patient = pd.DataFrame({
        'patient_id': [2],
        'age': [35],
        'gender': ['F'],
        'admission_type': ['Elective'],
        'num_medications': [2],
        'num_procedures': [1],
        'time_in_hospital': [2],
        'num_lab_procedures': [20],
        'num_diagnoses': [2],
        'previous_visits': [0],
        'diabetes': [0],
        'hypertension': [0],
        'heart_disease': [0]
    })
    
    for patient_name, patient_data in [("High-risk", high_risk_patient), ("Low-risk", low_risk_patient)]:
        X_patient, _ = preprocess_data(patient_data)
        
        # Ensure all features match
        for col in X.columns:
            if col not in X_patient.columns:
                X_patient[col] = 0
        X_patient = X_patient[X.columns]
        
        # Predict
        pred_proba = model.predict_proba(X_patient)[0, 1]
        risk_category = calculate_risk_score(pred_proba)
        
        print(f"\n   {patient_name} Patient:")
        print(f"   - Age: {patient_data['age'].values[0]}")
        print(f"   - Medications: {patient_data['num_medications'].values[0]}")
        print(f"   - Previous visits: {patient_data['previous_visits'].values[0]}")
        print(f"   - Readmission probability: {pred_proba:.1%}")
        print(f"   - Risk category: {risk_category}")
    
    # 6. Summary statistics
    print("\n6. Dataset summary statistics:")
    print(f"   Average age: {df['age'].mean():.1f} years")
    print(f"   Average medications: {df['num_medications'].mean():.1f}")
    print(f"   Average hospital stay: {df['time_in_hospital'].mean():.1f} days")
    print(f"   Patients with diabetes: {df['diabetes'].sum()} ({df['diabetes'].sum()/len(df):.1%})")
    print(f"   Emergency admissions: {(df['admission_type'] == 'Emergency').sum()} ({(df['admission_type'] == 'Emergency').sum()/len(df):.1%})")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nTo run the interactive web application, use:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
