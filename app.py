"""
Clinical Readmission Risk Explorer - Streamlit Web Application

A tool for exploring and predicting patient readmission risk using machine learning.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import (
    generate_sample_data, 
    preprocess_data, 
    calculate_risk_score,
    get_feature_importance_interpretation
)
from model import ReadmissionRiskModel


# Page configuration
st.set_page_config(
    page_title="Clinical Readmission Risk Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data(n_samples=1000):
    """Load or generate sample data."""
    return generate_sample_data(n_samples)


@st.cache_resource
def train_model(df):
    """Train the readmission risk model."""
    X, y = preprocess_data(df)
    model = ReadmissionRiskModel()
    metrics = model.train(X, y)
    return model, metrics


def main():
    """Main application function."""
    
    # Header
    st.title("🏥 Clinical Readmission Risk Explorer")
    
    st.markdown("""
    This tool uses machine learning to predict the risk of patient readmission to hospitals.
    Explore patient data, train models, and assess readmission risk for individual patients.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Dashboard", "Data Explorer", "Risk Prediction", "Model Analysis"]
    )
    
    # Load data
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Settings")
    n_samples = st.sidebar.slider("Number of patients", 100, 5000, 1000, 100)
    df = load_data(n_samples)
    
    # Train model
    model, metrics = train_model(df)
    
    # Display selected page
    if page == "Dashboard":
        show_dashboard(df, metrics)
    elif page == "Data Explorer":
        show_data_explorer(df)
    elif page == "Risk Prediction":
        show_risk_prediction(df, model)
    elif page == "Model Analysis":
        show_model_analysis(model, df)


def show_dashboard(df, metrics):
    """Display the main dashboard."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(df))
    
    with col2:
        readmission_rate = (df['readmitted'].sum() / len(df)) * 100
        st.metric("Readmission Rate", f"{readmission_rate:.1f}%")
    
    with col3:
        avg_age = df['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    
    with col4:
        st.metric("Model Accuracy", f"{metrics['accuracy']*100:.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Readmission Distribution")
        fig = px.pie(
            df, 
            names='readmitted',
            title='Patients by Readmission Status',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_traces(labels=['Not Readmitted', 'Readmitted'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution")
        fig = px.histogram(
            df,
            x='age',
            color='readmitted',
            title='Age Distribution by Readmission Status',
            barmode='overlay',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_traces(name='Not Readmitted', selector=dict(name='0'))
        fig.update_traces(name='Readmitted', selector=dict(name='1'))
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Admission Type Analysis")
        admission_data = df.groupby(['admission_type', 'readmitted']).size().reset_index(name='count')
        fig = px.bar(
            admission_data,
            x='admission_type',
            y='count',
            color='readmitted',
            title='Readmissions by Admission Type',
            barmode='group',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_traces(name='Not Readmitted', selector=dict(name='0'))
        fig.update_traces(name='Readmitted', selector=dict(name='1'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Hospital Stay Duration")
        fig = px.box(
            df,
            x='readmitted',
            y='time_in_hospital',
            title='Hospital Stay Duration by Readmission Status',
            color='readmitted',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        fig.update_xaxes(ticktext=['Not Readmitted', 'Readmitted'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


def show_data_explorer(df):
    """Display the data explorer page."""
    st.header("🔍 Data Explorer")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_range = st.slider("Age Range", int(df['age'].min()), int(df['age'].max()), 
                             (int(df['age'].min()), int(df['age'].max())))
    
    with col2:
        gender_filter = st.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    
    with col3:
        admission_filter = st.multiselect("Admission Type", df['admission_type'].unique(), 
                                          default=df['admission_type'].unique())
    
    # Apply filters
    filtered_df = df[
        (df['age'] >= age_range[0]) &
        (df['age'] <= age_range[1]) &
        (df['gender'].isin(gender_filter)) &
        (df['admission_type'].isin(admission_filter))
    ]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} patients")
    
    # Display data
    st.subheader("Patient Data")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="patient_data.csv",
        mime="text/csv"
    )
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(filtered_df.describe(), use_container_width=True)


def show_risk_prediction(df, model):
    """Display the risk prediction page."""
    st.header("🎯 Risk Prediction")
    
    st.markdown("""
    Enter patient information below to predict their readmission risk.
    The model will provide a risk score and category.
    """)
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=65)
        gender = st.selectbox("Gender", ["M", "F"])
        admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])
        num_medications = st.number_input("Number of Medications", min_value=0, max_value=50, value=5)
        num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=20, value=2)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=200, value=50)
    
    with col2:
        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=30, value=5)
        num_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=5)
        previous_visits = st.number_input("Previous Hospital Visits", min_value=0, max_value=20, value=1)
        diabetes = st.checkbox("Diabetes")
        hypertension = st.checkbox("Hypertension")
        heart_disease = st.checkbox("Heart Disease")
    
    # Predict button
    if st.button("Predict Readmission Risk", type="primary"):
        # Create patient data
        patient_data = pd.DataFrame({
            'patient_id': [999999],
            'age': [age],
            'gender': [gender],
            'admission_type': [admission_type],
            'num_medications': [num_medications],
            'num_procedures': [num_procedures],
            'time_in_hospital': [time_in_hospital],
            'num_lab_procedures': [num_lab_procedures],
            'num_diagnoses': [num_diagnoses],
            'previous_visits': [previous_visits],
            'diabetes': [int(diabetes)],
            'hypertension': [int(hypertension)],
            'heart_disease': [int(heart_disease)]
        })
        
        # Preprocess
        X_patient, _ = preprocess_data(patient_data)
        
        # Ensure all features are present
        X_train, _ = preprocess_data(df)
        for col in X_train.columns:
            if col not in X_patient.columns:
                X_patient[col] = 0
        X_patient = X_patient[X_train.columns]
        
        # Predict
        prediction_proba = model.predict_proba(X_patient)[0, 1]
        risk_category = calculate_risk_score(prediction_proba)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Readmission Probability", f"{prediction_proba*100:.1f}%")
        
        with col2:
            # Color-code risk category
            if risk_category == "Low Risk":
                st.success(f"Risk Category: {risk_category}")
            elif risk_category == "Medium Risk":
                st.warning(f"Risk Category: {risk_category}")
            else:
                st.error(f"Risk Category: {risk_category}")
        
        with col3:
            recommendation = "Standard care" if prediction_proba < 0.3 else \
                           "Enhanced monitoring" if prediction_proba < 0.6 else \
                           "Intensive follow-up"
            st.info(f"Recommendation: {recommendation}")
        
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Readmission Risk"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},
                    {'range': [30, 60], 'color': "#f39c12"},
                    {'range': [60, 100], 'color': "#e74c3c"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)


def show_model_analysis(model, df):
    """Display the model analysis page."""
    st.header("🔬 Model Analysis")
    
    # Model performance
    st.subheader("Model Performance")
    X, y = preprocess_data(df)
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = model.get_feature_importance()
    top_features = importance_df.head(10).copy()
    top_features['feature_label'] = top_features['feature'].apply(get_feature_importance_interpretation)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature_label',
        orientation='h',
        title='Top 10 Most Important Features',
        labels={'importance': 'Importance Score', 'feature_label': 'Feature'},
        color='importance',
        color_continuous_scale='blues'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance table
    with st.expander("View All Features"):
        importance_display = importance_df.copy()
        importance_display['feature'] = importance_display['feature'].apply(get_feature_importance_interpretation)
        st.dataframe(importance_display, use_container_width=True)
    
    # Model insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Characteristics:**
        - Algorithm: Random Forest Classifier
        - Features: 13+ clinical variables
        - Training approach: Balanced class weights
        - Validation: Train-test split (80-20)
        """)
    
    with col2:
        st.markdown("""
        **Top Risk Factors:**
        1. Previous hospital visits
        2. Number of medications
        3. Age of patient
        4. Length of hospital stay
        """)
    
    # Additional visualizations
    st.subheader("Risk Factor Analysis")
    
    selected_feature = st.selectbox(
        "Select feature to analyze",
        ['age', 'num_medications', 'time_in_hospital', 'previous_visits', 
         'num_procedures', 'num_diagnoses']
    )
    
    fig = px.violin(
        df,
        y=selected_feature,
        x='readmitted',
        box=True,
        title=f'{selected_feature.replace("_", " ").title()} Distribution by Readmission Status',
        color='readmitted',
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    fig.update_xaxes(ticktext=['Not Readmitted', 'Readmitted'], tickvals=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
