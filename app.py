from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open("model/best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Mapping for categorical features
housing_mapping = {
    "Rented apartment": 0,
    "House / apartment": 1,
    "Municipal apartment": 2,
    "Co-op apartment": 3,
    "Office apartment": 4
}

job_mapping = {
    "Working": 0,
    "Commercial associate": 1,
    "Pensioner": 2,
    "State servant": 3,
    "Unemployed": 4
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from the request
        data = request.json
        age = int(data["age"])
        income = float(data["income"])
        housing = data["housing"]
        family_members = int(data["family_members"])
        job_status = data["job_status"]

        # Convert categorical features
        housing_encoded = housing_mapping.get(housing, 0)
        job_encoded = job_mapping.get(job_status, 0)

        # Prepare input features
        input_features = np.array([[age, income, housing_encoded, family_members, job_encoded]])

        # Scale the input features
        input_scaled = scaler.transform(input_features)

        # Predict approval
        prediction = model.predict(input_scaled)[0]

        # Return JSON response
        return jsonify({"approved": bool(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
