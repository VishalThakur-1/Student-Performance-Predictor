from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    previous_marks = float(request.form["previous_marks"])
    assignments = float(request.form["assignments"])
    sleep_hours = float(request.form["sleep_hours"])

    input_data = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Previous_Marks": [previous_marks],
        "Assignments_Completed": [assignments],
        "Sleep_Hours": [sleep_hours]
    })

    prediction = model.predict(input_data)

    return render_template(
    "index.html",
    prediction=round(prediction[0],2)
)


app.run(debug=True)