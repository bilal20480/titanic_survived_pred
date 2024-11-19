import streamlit as st
import numpy as np
import pickle

# Load the trained model (make sure to save your trained model as 'logistic_model.pkl')
# This requires a saved pickle file containing your trained LogisticRegression model
# Uncomment the below two lines if you have the model saved
with open('titanic.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Diabetes Prediction App')

st.write("Enter the following details to predict diabetes:")

# Collecting input data from the user
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure Level', min_value=0, max_value=150, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin Level', min_value=0, max_value=800, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0, step=1)

# Making predictions based on input data
if st.button('Predict'):
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    
    # Uncomment when you have the trained model loaded
    # prediction = model.predict(input_data_reshape)
    
    # Placeholder for actual prediction result
    prediction = [0]  # Replace with the model prediction
    
    if prediction[0] == 0:
        st.write("The person is not diabetic.")
    else:
        st.write("The person is diabetic.")
