from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load dataset once when app starts
data = pd.read_csv("data/data.csv")

@app.route("/")
def home():
    # Show the form page
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # 1️⃣ Get user inputs from form
    budget = int(request.form.get("budget", 0))
    ram = int(request.form.get("ram", 0))
    usage = request.form.get("usage", "")

    # 2️⃣ Filter mobiles based on rules
    filtered_data = data[
        (data["price"] <= budget) &
        (data["ram"] >= ram) &
        (data["usage"] == usage)
    ]

    # 3️⃣ Add value-for-money score
    filtered_data["value_score"] = (
        filtered_data["rating"] / filtered_data["price"]
    )

    # 4️⃣ Sort by best value first
    filtered_data = filtered_data.sort_values(
        by="value_score",
        ascending=False
    )

    # 5️⃣ Convert to list of dictionaries
    mobiles = filtered_data.to_dict(orient="records")

    return render_template("results.html", mobiles=mobiles)

if __name__ == "__main__":
    app.run(debug=True)
