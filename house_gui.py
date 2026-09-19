import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset load
data = pd.read_csv("house_data.csv")

# Location convert
data["Location"] = data["Location"].map({
    "Jaipur": 1
})

# Features and target
X = data[["Area_sqft", "Bedrooms", "Bathrooms", "Location"]]
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)


# Prediction function
def predict_price():
    try:
        area = float(area_entry.get())
        bedrooms = int(bedroom_entry.get())
        bathrooms = int(bathroom_entry.get())

        if area <= 0 or bedrooms <= 0 or bathrooms <= 0:
            messagebox.showerror("Error", "Please enter positive values.")
            return

        # Jaipur = 1
        location = 1

        new_house = pd.DataFrame(
            [[area, bedrooms, bathrooms, location]],
            columns=["Area_sqft", "Bedrooms", "Bathrooms", "Location"]
        )

        predicted_price = model.predict(new_house)[0]

        result_label.config(
            text=f"Predicted House Price\n₹ {predicted_price:,.2f}"
        )

    except ValueError:
        messagebox.showerror(
            "Error",
            "Please enter valid numbers."
        )


# Clear function
def clear_fields():
    area_entry.delete(0, tk.END)
    bedroom_entry.delete(0, tk.END)
    bathroom_entry.delete(0, tk.END)

    result_label.config(
        text="Predicted House Price\n₹ 0.00"
    )


# Main window
root = tk.Tk()
root.title("House Price Prediction")
root.geometry("550x700")
root.resizable(False, False)

# Heading
title_label = tk.Label(
    root,
    text="HOUSE PRICE PREDICTION",
    font=("Arial", 22, "bold")
)
title_label.pack(pady=30)

subtitle = tk.Label(
    root,
    text="Enter House Details",
    font=("Arial", 14)
)
subtitle.pack(pady=10)

# Area
area_label = tk.Label(
    root,
    text="Area (sqft)",
    font=("Arial", 13)
)
area_label.pack(pady=(20, 5))

area_entry = tk.Entry(
    root,
    font=("Arial", 14),
    width=30
)
area_entry.pack()

# Bedrooms
bedroom_label = tk.Label(
    root,
    text="Number of Bedrooms",
    font=("Arial", 13)
)
bedroom_label.pack(pady=(20, 5))

bedroom_entry = tk.Entry(
    root,
    font=("Arial", 14),
    width=30
)
bedroom_entry.pack()

# Bathrooms
bathroom_label = tk.Label(
    root,
    text="Number of Bathrooms",
    font=("Arial", 13)
)
bathroom_label.pack(pady=(20, 5))

bathroom_entry = tk.Entry(
    root,
    font=("Arial", 14),
    width=30
)
bathroom_entry.pack()

# Predict button
predict_button = tk.Button(
    root,
    text="PREDICT PRICE",
    font=("Arial", 14, "bold"),
    command=predict_price,
    width=20,
    height=2
)
predict_button.pack(pady=30)

# Clear button
clear_button = tk.Button(
    root,
    text="CLEAR",
    font=("Arial", 11),
    command=clear_fields,
    width=15
)
clear_button.pack()

# Result
result_label = tk.Label(
    root,
    text="Predicted House Price\n₹ 0.00",
    font=("Arial", 18, "bold")
)
result_label.pack(pady=20)

# Start application
root.mainloop()