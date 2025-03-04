import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load trained model
loaded_model = pickle.load(open(r"C:\Users\ANIRBAN GHOSH\Downloads\Obesity\trained_model1.sav", 'rb'))
scaler = pickle.load(open(r"C:\Users\ANIRBAN GHOSH\Downloads\Obesity\scaler.sav", 'rb'))

# 🔹 Convert input data into numeric format
def input_conversion(a):
    mapping = {
        "male": 0, "female": 1,
        "yes": 0, "no": 1,
        "never": 0, "sometimes": 1, "frequently": 2, "always": 3,
        "automobile": 0, "bike": 1, "motorbike": 2, "public transportation": 3, "walking": 4
    }
    
    # Convert all string values to lowercase and map them
    for i in range(len(a)):
        if isinstance(a[i], str):  
            a[i] = mapping.get(a[i].lower(), -1)  # -1 for unknown values

    return a

# 🔹 Prediction function
def obesity_prediction(input_data):
    numeric_data = input_conversion(input_data)

    # Convert to NumPy array
    input_data_as_numpy_array = np.asarray(numeric_data, dtype=float).reshape(1, -1)

    # Normalize Input
    input_data_scaled = scaler.transform(input_data_as_numpy_array)

    # Make prediction
    prediction = loaded_model.predict(input_data_scaled)

    # Interpretation
    obesity_messages = {
        0: "You are underweight. Consider a balanced diet to gain healthy weight.",
        1: "You have a healthy weight. Keep maintaining a balanced lifestyle!",
        2: "You are slightly overweight(Overweight_Level_I). Regular exercise and a healthy diet are recommended.",
        3: "You are overweight(Overweight_Level_II). Consider consulting a dietitian for a healthier routine.",
        4: "You are in Obesity Type I. Prioritize physical activity and healthy eating habits.",
        5: "You are in Obesity Type II. Seek medical advice to prevent health risks.",
        6: "You are in Obesity Type III (Severe Obesity). Immediate medical intervention is recommended."
    }

    return obesity_messages.get(prediction[0], "Invalid category. Please check your input.")

# 🔹 Streamlit App
def main():
    st.title("Obesity Prediction Web App")

    # Input fields
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', min_value=1, max_value=100)
    Height = st.number_input('Height (meters)', min_value=0.5, max_value=3.0, step=0.01)
    Weight = st.number_input('Weight (kg)', min_value=10.0, max_value=300.0, step=0.1)
    family_history_with_overweight = st.selectbox('Family History of Overweight', ['Yes', 'No'])
    FAVC = st.selectbox('Frequently Consumes High-Calorie Foods?', ['Yes', 'No'])
    FCVC = st.slider('Frequency of Vegetable Consumption (1-3)', 1, 3)
    NCP = st.slider('Number of Main Meals Per Day', 1, 6)
    CAEC = st.selectbox('Consumes Food Between Meals?', ['Never', 'Sometimes', 'Frequently', 'Always'])
    SMOKE = st.selectbox('Do You Smoke?', ['Yes', 'No'])
    CH2O = st.slider('Daily Water Intake (1-3)', 1, 3)
    SCC = st.selectbox('Do You Monitor Calorie Intake?', ['Yes', 'No'])
    FAF = st.slider('Physical Activity Frequency (0-3)', 0, 3)
    TUE = st.slider('Usage of Electronic Devices (0-3)', 0, 3)
    CALC = st.selectbox('Alcohol Consumption?', ['Never', 'Sometimes', 'Frequently', 'Always'])
    MTRANS = st.selectbox('Main Mode of Transportation', ['Automobile', 'Bike', 'Motorbike', 'Public Transportation', 'Walking'])

    # Predict Button
    if st.button('Predict Obesity Level'):
        Result = obesity_prediction([Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS])
        st.success(Result)

if __name__ == '__main__':
    main()
