from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Dataset load
data = pd.read_csv("house_data.csv")

# Location encoding
data["Location"] = data["Location"].map({
    "Jaipur": 1
})

# Features and target
X = data[["Area_sqft", "Bedrooms", "Bathrooms", "Location"]]
y = data["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        location = request.form["location"]

        location_code = 1 if location == "Jaipur" else 1

        new_house = pd.DataFrame(
            [[area, bedrooms, bathrooms, location_code]],
            columns=["Area_sqft", "Bedrooms", "Bathrooms", "Location"]
        )

        predicted_price = model.predict(new_house)[0]

        prediction = f"₹ {predicted_price:,.2f}"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)