
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load trained model
loaded_model = pickle.load(open(r"C:\Users\ANIRBAN GHOSH\Downloads\Obesity\trained_model1.sav", 'rb'))
scaler = pickle.load(open(r"C:\Users\ANIRBAN GHOSH\Downloads\Obesity\scaler.sav", 'rb'))
# Sample input
input_data = ["Male", 24, 1.78, 64.00, "yes", "no", 3.00, 3.00, "Sometimes", "no", 2.00, "no", 1.00, 1.00, "Frequently", "Public_Transportation"]

def input_conversion(a):
    mapping = {
        "Male": 0, "Female": 1,
        "yes": 0, "no": 1,
        "Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3,
        "Automobile": 0, "Bike": 1, "Motorbike": 2, "Public_Transportation": 3, "Walking": 4
    }
    
    for i in range(len(a)):
        if a[i] in mapping:
            a[i] = mapping[a[i]]
    
    return a

# Convert input
numeric_data = input_conversion(input_data)
print("Encoded Input:", numeric_data)  # Debugging Step

# Convert to NumPy array
input_data_as_numpy_array = np.asarray(numeric_data).reshape(1, -1)
# Normalize Input (if trained on scaled data)
input_data_as_numpy_array = scaler.transform(input_data_as_numpy_array)
std_data = scaler.transform(input_data_as_numpy_array)
print(std_data)

# Make prediction
prediction = loaded_model.predict(input_data_as_numpy_array)
print("Predicted Label:", prediction)

# Interpretation
obesity_messages = {
    0: "You are underweight. Consider a balanced diet to gain healthy weight.",
    1: "You have a healthy weight. Keep maintaining a balanced lifestyle!",
    2: "You are slightly overweight. Regular exercise and a healthy diet are recommended.",
    3: "You are overweight. Consider consulting a dietitian for a healthier routine.",
    4: "You are in Obesity Type I. Prioritize physical activity and healthy eating habits.",
    5: "You are in Obesity Type II. Seek medical advice to prevent health risks.",
    6: "You are in Obesity Type III (Severe Obesity). Immediate medical intervention is recommended."
}

print(obesity_messages.get(prediction[0], "Invalid category. Please check your input."))
