import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Smart Health Risk Prediction System")

# Form fields (match your features exactly)
age = st.number_input('Age', 18, 100, 30)
cholesterol = st.slider('Cholesterol', 100, 300, 200)
heart_rate = st.slider('Heart Rate', 40, 150, 70)
alcohol = st.slider('Alcohol Consumption', 0, 15, 0)
exercise = st.slider('Exercise Hours Per Week', 0, 21, 3)
stress = st.slider('Stress Level', 0, 10, 5)
sedentary = st.slider('Sedentary Hours Per Day', 0, 16, 8)
income = st.number_input('Income', 0, 1000000, 50000)
bmi = st.slider('BMI', 10, 50, 25)
triglycerides = st.slider('Triglycerides', 50, 300, 150)
physical_activity_days = st.slider('Physical Activity Days Per Week', 0, 7, 4)
sleep_hours = st.slider('Sleep Hours Per Day', 4, 12, 7)
systolic_bp = st.slider('Systolic BP', 80, 200, 120)
diastolic_bp = st.slider('Diastolic BP', 40, 130, 80)

sex = st.selectbox('Sex', ['Male', 'Female'])
diabetes = st.selectbox('Diabetes', ['No', 'Yes'])
family_history = st.selectbox('Family History', ['No', 'Yes'])
smoking = st.selectbox('Smoking', ['No', 'Yes'])
obesity = st.selectbox('Obesity', ['No', 'Yes'])
diet = st.selectbox('Diet', ['Unhealthy', 'Average', 'Healthy'])
previous_heart = st.selectbox('Previous Heart Problems', ['No', 'Yes'])
medication = st.selectbox('Medication Use', ['No', 'Yes'])
country = st.selectbox('Country', ['India','Japan','USA','Canada','Nigeria'])  # Update with actual label-encoded mapping
continent = st.selectbox('Continent', ['Asia','North America','Africa'])      # Update with actual mapping
hemisphere = st.selectbox('Hemisphere', ['Northern Hemisphere','Southern Hemisphere'])  # Mapping needed

# Convert the categorical input to match label encoding (0/1/2)
sex = 1 if sex == 'Male' else 0
diabetes = 1 if diabetes == 'Yes' else 0
family_history = 1 if family_history == 'Yes' else 0
smoking = 1 if smoking == 'Yes' else 0
obesity = 1 if obesity == 'Yes' else 0
diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2}
diet = diet_map[diet]
previous_heart = 1 if previous_heart == 'Yes' else 0
medication = 1 if medication == 'Yes' else 0
# You must ensure country/continent/hemisphere inputs align with your training label encoding!

# Prepare the input in DataFrame
input_df = pd.DataFrame([{
    'Sex': sex, 'Diabetes': diabetes, 'Family History': family_history, 'Smoking': smoking,
    'Obesity': obesity, 'Diet': diet, 'Previous Heart Problems': previous_heart,
    'Medication Use': medication, 'Country': 0, 'Continent': 0, 'Hemisphere': 0,  # Update as above
    'Age': age, 'Cholesterol': cholesterol, 'Heart Rate': heart_rate, 'Alcohol Consumption': alcohol,
    'Exercise Hours Per Week': exercise, 'Stress Level': stress, 'Sedentary Hours Per Day': sedentary,
    'Income': income, 'BMI': bmi, 'Triglycerides': triglycerides,
    'Physical Activity Days Per Week': physical_activity_days, 'Sleep Hours Per Day': sleep_hours,
    'Systolic BP': systolic_bp, 'Diastolic BP': diastolic_bp
}])

# Scale only the numeric columns
num_cols = ['Age', 'Cholesterol', 'Heart Rate', 'Alcohol Consumption',
            'Exercise Hours Per Week', 'Stress Level', 'Sedentary Hours Per Day', 'Income',
            'BMI', 'Triglycerides', 'Physical Activity Days Per Week', 'Sleep Hours Per Day',
            'Systolic BP', 'Diastolic BP']
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Predict button
if st.button('Predict Risk'):
    risk = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"Predicted Heart Attack Risk: {risk} (Probability: {prob:.2f})")
