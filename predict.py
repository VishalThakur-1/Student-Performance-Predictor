import pickle
import pandas as pd

# Load trained model
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

# User input
study_hours = float(input("Enter Study Hours: "))
attendance = float(input("Enter Attendance: "))
previous_marks = float(input("Enter Previous Marks: "))
assignments = float(input("Enter Assignments Completed: "))
sleep_hours = float(input("Enter Sleep Hours: "))

# Prediction
input_data = pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance": [attendance],
    "Previous_Marks": [previous_marks],
    "Assignments_Completed": [assignments],
    "Sleep_Hours": [sleep_hours]
})

prediction = model.predict(input_data)

# Display result
print(f"\n🎯 Predicted Final Marks: {prediction[0]:.2f}")