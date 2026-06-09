# Imports
import joblib

# Load the model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

"""
Sample inputs
Congratulations you have won a free phone. Contact 111-222-333 to get it.
Hello, can we schedule our meeting next Monday ?
"""

# Get the input
message = input("Enter message: ")

# Convert text to vector
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)[0]

# Output
if prediction == "spam":
    print("Spam message")
else:
    print("Not a spam message")
