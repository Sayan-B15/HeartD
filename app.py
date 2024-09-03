import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
try:
    model = joblib.load('HeartDiseaseDetection')
    # Print and store model features
    feature_names = model.feature_names_in_
    print("Model features:", feature_names)
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

def predict_heart_disease(input_data):
    # Create a DataFrame with the input data, ensuring correct feature order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Make prediction
    try:
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def main():
    st.title('Heart Disease Prediction App')

    st.write('Please enter the following information:')

    # Input fields
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=1000, value=200)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=300, value=120)
    heart_rate = st.number_input('Heart Rate', min_value=0, max_value=220, value=70)
    smoking = st.selectbox('Smoking', ['Never', 'Former', 'Current'])
    alcohol_intake = st.selectbox('Alcohol Intake', ['Never', 'Moderate', 'Heavy'])
    exercise_hours = st.number_input('Exercise Hours', min_value=0, max_value=24, value=1)
    family_history = st.selectbox('Family History', ['Yes', 'No'])
    diabetes = st.selectbox('Diabetes', ['Yes', 'No'])
    obesity = st.selectbox('Obesity', ['Yes', 'No'])
    stress_level = st.slider('Stress Level', min_value=0, max_value=10, value=5)
    blood_sugar = st.number_input('Blood Sugar', min_value=0, max_value=500, value=100)
    exercise_angina = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])

    if st.button('Predict'):
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

        # Ensure all features are present, even if they're not used in input
        for feature in feature_names:
            if feature not in input_data:
                input_data[feature] = 0

        prediction = predict_heart_disease(input_data)

        if prediction is not None:
            st.subheader('Prediction Results:')
            if prediction == 1:
                st.write('The model predicts that you may have a heart disease.')
            else:
                st.write('The model predicts that you may not have a heart disease.')
            
            st.write("Note: This model does not provide probability estimates.")
        else:
            st.error("Unable to make a prediction. Please check your inputs and try again.")

if __name__ == '__main__':
    main()