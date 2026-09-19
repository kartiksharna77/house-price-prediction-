from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# 1. LOAD DATASET
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(BASE_DIR, "house_data.csv"))

print("\n================ HOUSE PRICE DATASET ================\n")
print(data)


# =========================================================
# 2. SELECT FEATURES AND TARGET
# =========================================================

X = data[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Location"
    ]
]

y = data["Price"]


# =========================================================
# 3. CONVERT LOCATION INTO NUMERIC CODE
# =========================================================

location_codes = {
    "Jaipur": 1,
    "Delhi": 2,
    "Mumbai": 3,
    "Bangalore": 4,
    "Pune": 5
}

X = X.copy()

X["Location"] = X["Location"].map(location_codes)

# Missing location ko 0 kar do
X["Location"] = X["Location"].fillna(0)


# =========================================================
# 4. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 5. TRAIN LINEAR REGRESSION MODEL
# =========================================================

model = LinearRegression()

model.fit(X_train, y_train)


# =========================================================
# 6. MODEL PREDICTION
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 7. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\n================ MODEL RESULT ================\n")

print(f"Mean Absolute Error: {mae:.2f}")

print(f"R2 Score: {r2:.4f}")


# =========================================================
# 8. HOME PAGE
# =========================================================

HTML_PAGE = """
<!DOCTYPE html>

<html>

<head>

    <title>House Price Prediction</title>

    <style>

        body {
            font-family: Arial, sans-serif;
            background: #f2f2f2;
            margin: 0;
            padding: 0;
        }

        .container {
            width: 500px;
            margin: 60px auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.2);
        }

        h1 {
            text-align: center;
            color: #333;
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input, select {
            width: 100%;
            padding: 12px;
            margin-top: 5px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 8px;
        }

        button {
            width: 100%;
            padding: 13px;
            margin-top: 25px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #0056b3;
        }

        .result {
            margin-top: 25px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 8px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }

        .info {
            margin-top: 25px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>🏠 House Price Prediction</h1>

    <form method="POST">

        <label>Area (sqft)</label>

        <input
            type="number"
            name="area"
            placeholder="Enter area"
            required
        >


        <label>Number of Bedrooms</label>

        <input
            type="number"
            name="bedrooms"
            placeholder="Enter bedrooms"
            required
        >


        <label>Number of Bathrooms</label>

        <input
            type="number"
            name="bathrooms"
            placeholder="Enter bathrooms"
            required
        >


        <label>Location</label>

        <select name="location" required>

            <option value="Jaipur">Jaipur</option>

            <option value="Delhi">Delhi</option>

            <option value="Mumbai">Mumbai</option>

            <option value="Bangalore">Bangalore</option>

            <option value="Pune">Pune</option>

        </select>


        <button type="submit">
            Predict House Price
        </button>

    </form>


    {% if prediction %}

    <div class="result">

        Predicted House Price:

        <br>

        ₹ {{ prediction }}

    </div>

    {% endif %}


    <div class="info">

        <b>Model Information</b>

        <br><br>

        Algorithm: Linear Regression

        <br>

        Mean Absolute Error: {{ mae }}

        <br>

        R2 Score: {{ r2 }}

    </div>

</div>

</body>

</html>
"""


# =========================================================
# 9. FLASK ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])

def home():

    prediction = None

    if request.method == "POST":

        # User input
        area = float(request.form["area"])

        bedrooms = int(request.form["bedrooms"])

        bathrooms = int(request.form["bathrooms"])

        location = request.form["location"]


        # Location code
        location_code = location_codes.get(location, 0)


        # Create input dataframe
        new_house = pd.DataFrame(
            [[
                area,
                bedrooms,
                bathrooms,
                location_code
            ]],

            columns=[
                "Area_sqft",
                "Bedrooms",
                "Bathrooms",
                "Location"
            ]
        )


        # Prediction
        predicted_price = model.predict(new_house)[0]


        # Format price
        prediction = f"{predicted_price:,.2f}"


    return render_template_string(
        HTML_PAGE,

        prediction=prediction,

        mae=f"{mae:,.2f}",

        r2=f"{r2:.4f}"
    )


# =========================================================
# 10. RUN FLASK APPLICATION
# =========================================================

if __name__ == "__main__":

    print("\n==========================================")

    print("HOUSE PRICE PREDICTION WEB APP")

    print("==========================================")

    print("\nOpen this URL in your browser:")

    print("http://127.0.0.1:5000")

    print("\n==========================================\n")

    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=False)
      
    
