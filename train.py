import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# Load dataset
df = pd.read_csv("data/student-dataset.csv")

# Features and Target
X = df.drop("Final_Marks", axis=1)
y = df["Final_Marks"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("🎉 Model trained successfully!")


# Save the trained model
with open("models/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved successfully!")


# Predict on test data
predictions = model.predict(X_test)

print("Predicted Marks:")
print(predictions)

print("\nActual Marks:")
print(y_test.values)


#Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

#Mean Squared Error

mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error:", mse)

#R² Score


r2 = r2_score(y_test, predictions)

print("R² Score:", r2)