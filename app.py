import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

# Custom CSS to improve app appearance
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 18px;
    }
    .risk-factor {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .high-risk {
        background-color: #ffcccb;
    }
    .moderate-risk {
        background-color: #fffacd;
    }
    .low-risk {
        background-color: #90ee90;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('HeartDiseaseDetection')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

model = load_model()

if model is None:
    st.stop()

# Print and store model features
feature_names = model.feature_names_in_
print("Model features:", feature_names)

def predict_heart_disease(input_data):
    input_df = pd.DataFrame([input_data], columns=feature_names)
    try:
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title('❤️ Heart Disease Prediction App')
    st.write('This app predicts the likelihood of heart disease based on your input.')

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.number_input('Age', min_value=0, max_value=120, value=30)
        gender = st.selectbox('Gender', ['Male', 'Female'])
        family_history = st.selectbox('Family History of Heart Disease', ['No', 'Yes'])
        
        st.subheader("Lifestyle Factors")
        smoking = st.selectbox('Smoking', ['Never', 'Former', 'Current'])
        alcohol_intake = st.selectbox('Alcohol Intake', ['Never', 'Moderate', 'Heavy'])
        exercise_hours = st.number_input('Exercise Hours per Week', min_value=0, max_value=168, value=7)
        stress_level = st.slider('Stress Level', min_value=0, max_value=10, value=5)

    with col2:
        st.subheader("Health Metrics")
        cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=0, max_value=1000, value=200)
        blood_pressure = st.number_input('Systolic Blood Pressure (mmHg)', min_value=0, max_value=300, value=120)
        heart_rate = st.number_input('Resting Heart Rate (bpm)', min_value=0, max_value=220, value=70)
        blood_sugar = st.number_input('Fasting Blood Sugar (mg/dL)', min_value=0, max_value=500, value=100)
        
        st.subheader("Medical Conditions")
        diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
        obesity = st.selectbox('Obesity', ['No', 'Yes'])
        exercise_angina = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
        chest_pain = st.selectbox('Chest Pain Type', ['None', 'Typical Angina', 'Atypical Angina', 'Non-anginal Pain'])

    if st.button('Predict Heart Disease Risk'):
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': 0 if gender == 'Female' else 1,
            'Cholesterol': cholesterol,
            'Blood Pressure': blood_pressure,
            'Heart Rate': heart_rate,
            'Exercise Hours': exercise_hours,
            'Family History': 1 if family_history == 'Yes' else 0,
            'Diabetes': 1 if diabetes == 'Yes' else 0,
            'Obesity': 1 if obesity == 'Yes' else 0,
            'Stress Level': stress_level,
            'Blood Sugar': blood_sugar,
            'Exercise Induced Angina': 1 if exercise_angina == 'Yes' else 0,
            'Former': 1 if smoking == 'Former' else 0,
            'Current': 1 if smoking == 'Current' else 0,
            'Moderate': 1 if alcohol_intake == 'Moderate' else 0,
            'Heavy': 1 if alcohol_intake == 'Heavy' else 0,
            'Typical Angina': 1 if chest_pain == 'Typical Angina' else 0,
            'Atypical Angina': 1 if chest_pain == 'Atypical Angina' else 0,
            'Non-anginal Pain': 1 if chest_pain == 'Non-anginal Pain' else 0
        }

        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = 0

        prediction = predict_heart_disease(input_data)

        if prediction is not None:
            st.subheader('Prediction Results:')
            if prediction == 1:
                st.error('⚠️ The model predicts that you may have a higher risk of heart disease.')
            else:
                st.success('✅ The model predicts that you may have a lower risk of heart disease.')
            
            st.info("Note: This prediction is based on the provided information and should not be considered a medical diagnosis. Please consult with a healthcare professional for proper evaluation.")

            # Display risk factors
            st.subheader("Risk Factor Analysis")
            risk_factors = []
            if age > 55:
                risk_factors.append(("Age > 55", "high-risk"))
            if gender == "Male":
                risk_factors.append(("Male gender", "moderate-risk"))
            if cholesterol > 200:
                risk_factors.append(("High cholesterol", "high-risk"))
            if blood_pressure > 140:
                risk_factors.append(("High blood pressure", "high-risk"))
            if smoking != "Never":
                risk_factors.append(("Smoking", "high-risk"))
            if alcohol_intake == "Heavy":
                risk_factors.append(("Heavy alcohol consumption", "high-risk"))
            if exercise_hours < 2.5:
                risk_factors.append(("Insufficient exercise", "moderate-risk"))
            if family_history == "Yes":
                risk_factors.append(("Family history of heart disease", "high-risk"))
            if diabetes == "Yes":
                risk_factors.append(("Diabetes", "high-risk"))
            if obesity == "Yes":
                risk_factors.append(("Obesity", "high-risk"))
            if stress_level > 7:
                risk_factors.append(("High stress level", "moderate-risk"))

            for factor, risk_level in risk_factors:
                st.markdown(f'<div class="risk-factor {risk_level}">{factor}</div>', unsafe_allow_html=True)

            if not risk_factors:
                st.markdown('<div class="risk-factor low-risk">No significant risk factors identified</div>', unsafe_allow_html=True)

            # Visualize some key metrics
            st.subheader("Key Health Metrics")
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

            ax1.pie([cholesterol, 200-cholesterol], labels=['Your Cholesterol', 'Ideal Range'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax1.set_title('Cholesterol Level')

            ax2.pie([blood_pressure, 120-blood_pressure], labels=['Your Blood Pressure', 'Ideal Range'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax2.set_title('Blood Pressure')

            ax3.pie([blood_sugar, 100-blood_sugar], labels=['Your Blood Sugar', 'Ideal Range'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax3.set_title('Blood Sugar')

            st.pyplot(fig)

        else:
            st.error("Unable to make a prediction. Please check your inputs and try again.")

if __name__ == '__main__':
    main()