from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD DATASET
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "house_data.csv")

data = pd.read_csv(DATA_PATH)


# ============================================================
# SELECT FEATURES AND TARGET
# ============================================================

X = data[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Location"
    ]
]

y = data["Price"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            ["Location"]
        )
    ],
    remainder="passthrough"
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ============================================================
# HTML PAGE
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <title>House Price Prediction</title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background: linear-gradient(
                135deg,
                #667eea,
                #764ba2
            );
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 95%;
            max-width: 650px;
            background: white;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.25);
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
            color: #333;
        }

        .subtitle {
            text-align: center;
            color: #777;
            margin-bottom: 30px;
        }

        label {
            display: block;
            margin-top: 18px;
            margin-bottom: 7px;
            font-weight: bold;
            color: #333;
        }

        input,
        select {
            width: 100%;
            padding: 13px;
            border: 1px solid #ccc;
            border-radius: 8px;
            font-size: 16px;
        }

        input:focus,
        select:focus {
            outline: none;
            border-color: #667eea;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 14px;
            border: none;
            border-radius: 8px;
            background: #667eea;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #5568d9;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            background: #f0f7ff;
            text-align: center;
        }

        .result-title {
            font-size: 16px;
            color: #555;
        }

        .price {
            font-size: 30px;
            font-weight: bold;
            color: #2e7d32;
            margin-top: 8px;
        }

        .model-info {
            margin-top: 25px;
            padding: 18px;
            background: #f7f7f7;
            border-radius: 10px;
        }

        .model-info h3 {
            margin-top: 0;
            color: #333;
        }

        .model-info p {
            margin: 8px 0;
            color: #555;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            background: #ffecec;
            color: #c62828;
            border-radius: 8px;
            text-align: center;
        }

        .footer {
            margin-top: 20px;
            text-align: center;
            color: #999;
            font-size: 13px;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>🏠 House Price Prediction</h1>

    <div class="subtitle">
        Predict house prices using Machine Learning
    </div>


    <form method="POST">

        <label for="area">
            Area (sqft)
        </label>

        <input
            type="number"
            id="area"
            name="area"
            placeholder="Example: 2000"
            min="100"
            step="1"
            required
            value="{{ area }}"
        >


        <label for="bedrooms">
            Number of Bedrooms
        </label>

        <input
            type="number"
            id="bedrooms"
            name="bedrooms"
            placeholder="Example: 4"
            min="1"
            step="1"
            required
            value="{{ bedrooms }}"
        >


        <label for="bathrooms">
            Number of Bathrooms
        </label>

        <input
            type="number"
            id="bathrooms"
            name="bathrooms"
            placeholder="Example: 3"
            min="1"
            step="1"
            required
            value="{{ bathrooms }}"
        >


        <label for="location">
            Location
        </label>

        <select
            id="location"
            name="location"
            required
        >

            {% for loc in locations %}

                <option
                    value="{{ loc }}"
                    {% if loc == location %}
                    selected
                    {% endif %}
                >
                    {{ loc }}
                </option>

            {% endfor %}

        </select>


        <button type="submit">
            Predict House Price
        </button>

    </form>


    {% if prediction %}

    <div class="result">

        <div class="result-title">
            Estimated House Price
        </div>

        <div class="price">
            {{ prediction }}
        </div>

    </div>

    {% endif %}


    {% if error %}

    <div class="error">
        {{ error }}
    </div>

    {% endif %}


    <div class="model-info">

        <h3>📊 Model Information</h3>

        <p>
            <strong>Algorithm:</strong>
            Linear Regression
        </p>

        <p>
            <strong>MAE:</strong>
            ₹{{ "{:,.2f}".format(mae) }}
        </p>

        <p>
            <strong>R² Score:</strong>
            {{ "{:.4f}".format(r2) }}
        </p>

        <p>
            <strong>Features:</strong>
            Area, Bedrooms, Bathrooms, Location
        </p>

    </div>


    <div class="footer">
        House Price Prediction using Machine Learning
    </div>

</div>

</body>

</html>
"""


# ============================================================
# HOME ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    area = ""
    bedrooms = ""
    bathrooms = ""

    locations = sorted(
        data["Location"].dropna().unique().tolist()
    )

    location = locations[0] if locations else ""

    if request.method == "POST":

        try:

            # --------------------------------------------
            # GET USER INPUT
            # --------------------------------------------

            area = float(
                request.form.get("area", 0)
            )

            bedrooms = int(
                request.form.get("bedrooms", 0)
            )

            bathrooms = int(
                request.form.get("bathrooms", 0)
            )

            location = request.form.get(
                "location",
                ""
            ).strip()


            # --------------------------------------------
            # VALIDATION
            # --------------------------------------------

            if area <= 0:
                raise ValueError(
                    "Area must be greater than 0."
                )

            if bedrooms <= 0:
                raise ValueError(
                    "Bedrooms must be greater than 0."
                )

            if bathrooms <= 0:
                raise ValueError(
                    "Bathrooms must be greater than 0."
                )

            if location not in locations:
                raise ValueError(
                    "Please select a valid location."
                )


            # --------------------------------------------
            # CREATE NEW HOUSE DATA
            # --------------------------------------------

            new_house = pd.DataFrame(
                [
                    {
                        "Area_sqft": area,
                        "Bedrooms": bedrooms,
                        "Bathrooms": bathrooms,
                        "Location": location
                    }
                ]
            )


            # --------------------------------------------
            # PREDICTION
            # --------------------------------------------

            predicted_price = model.predict(
                new_house
            )[0]


            # --------------------------------------------
            # SAFETY CHECK
            # --------------------------------------------

            if not np.isfinite(predicted_price):
                raise ValueError(
                    "Unable to calculate prediction."
                )


            # --------------------------------------------
            # FORMAT PRICE
            # --------------------------------------------

            prediction = (
                f"₹{predicted_price:,.2f}"
            )


        except Exception as e:

            error = str(e)


    return render_template_string(
        HTML,
        prediction=prediction,
        error=error,
        area=area,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        location=location,
        locations=locations,
        mae=mae,
        r2=r2
    )


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
