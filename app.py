import streamlit as st
import numpy as np
import pickle

# Load the trained model (ensure the model file is in the same directory or provide a path)
with open('titanic_pred.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Titanic Survival Prediction")

# User input fields for the features
pclass = st.selectbox("Passenger Class (Pclass)", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
sex = st.selectbox("Gender (0 = Male, 1 = Female)", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
age = st.number_input("Age", min_value=0, max_value=100, value=22)
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, value=1)
parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Passenger Fare", min_value=0.0, value=7.25)
embarked = st.selectbox("Port of Embarkation (0 = S, 1 = C, 2 = Q)", options=[0, 1, 2], format_func=lambda x: "S" if x == 0 else "C" if x == 1 else "Q")

# Input data conversion to numpy array and reshaping
input_data = (pclass, sex, age, sibsp, parch, fare, embarked)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data_reshape)
    if prediction[0] == 0:
        st.write("Prediction: Died")
    else:
        st.write("Prediction: Survived")
