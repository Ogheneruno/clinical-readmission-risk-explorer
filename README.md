# 🏥 Clinical Readmission Risk Explorer

A machine learning-powered web application for exploring and predicting patient hospital readmission risk. This tool helps healthcare professionals analyze patient data and assess the likelihood of readmission using clinical features.

## Features

- **Interactive Dashboard**: Visualize patient data and readmission statistics
- **Data Explorer**: Filter and analyze patient records with customizable views
- **Risk Prediction**: Predict individual patient readmission risk using machine learning
- **Model Analysis**: Examine feature importance and model performance metrics

## Technologies

- **Python 3.8+**: Core programming language
- **Streamlit**: Interactive web application framework
- **Scikit-learn**: Machine learning model (Random Forest Classifier)
- **Pandas & NumPy**: Data processing and analysis
- **Plotly & Matplotlib**: Data visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ogheneruno/clinical-readmission-risk-explorer.git
cd clinical-readmission-risk-explorer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Application Pages

### 1. Dashboard
- Overview of key metrics (total patients, readmission rate, average age, model accuracy)
- Visualizations of readmission distribution, age analysis, and admission types
- Hospital stay duration analysis

### 2. Data Explorer
- Interactive filtering by age, gender, and admission type
- Downloadable patient data in CSV format
- Statistical summaries and descriptive statistics

### 3. Risk Prediction
- Input form for patient clinical features:
  - Demographics (age, gender)
  - Admission details (type, duration)
  - Medical history (medications, procedures, diagnoses)
  - Chronic conditions (diabetes, hypertension, heart disease)
- Real-time risk prediction with probability score
- Risk categorization (Low/Medium/High)
- Visual risk gauge and recommendations

### 4. Model Analysis
- Feature importance visualization
- Top risk factors identification
- Model performance metrics
- Risk factor distribution analysis

## Model Details

The application uses a **Random Forest Classifier** trained on clinical features:

**Key Features:**
- Patient demographics (age, gender)
- Admission characteristics (type, duration)
- Clinical metrics (medications, procedures, lab tests, diagnoses)
- Medical history (previous visits, chronic conditions)

**Model Performance:**
- Balanced class weights to handle class imbalance
- 80-20 train-test split for validation
- Feature importance analysis for interpretability

## Sample Data

The application generates synthetic patient data for demonstration purposes. In a production environment, this would be replaced with real clinical data from hospital systems.

**Data Fields:**
- `patient_id`: Unique patient identifier
- `age`: Patient age (18-95 years)
- `gender`: Patient gender (M/F)
- `admission_type`: Type of admission (Emergency/Urgent/Elective)
- `num_medications`: Number of medications prescribed
- `num_procedures`: Number of procedures performed
- `time_in_hospital`: Length of hospital stay in days
- `num_lab_procedures`: Number of laboratory procedures
- `num_diagnoses`: Number of diagnoses recorded
- `previous_visits`: Number of previous hospital visits
- `diabetes`: Diabetes diagnosis (0/1)
- `hypertension`: Hypertension diagnosis (0/1)
- `heart_disease`: Heart disease diagnosis (0/1)
- `readmitted`: Readmission status (target variable, 0/1)

## Project Structure

```
clinical-readmission-risk-explorer/
├── app.py                  # Main Streamlit application
├── model.py                # Machine learning model implementation
├── data_processing.py      # Data processing utilities
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## Future Enhancements

- Integration with real hospital data sources (FHIR, HL7)
- Advanced ML models (XGBoost, Neural Networks)
- SHAP values for individual prediction explanations
- Patient risk stratification and cohort analysis
- API endpoint for programmatic access
- User authentication and role-based access control
- Export functionality for reports and dashboards

## Disclaimer

This application is for educational and demonstration purposes only. It should not be used for actual clinical decision-making without proper validation, regulatory approval, and integration with certified healthcare systems.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please open an issue on GitHub.
