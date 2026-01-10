from flask import Flask,render_template,request
import pandas as pd

app= Flask(__name__)

data = pd.read_csv("data/data.csv")
@app.route("/")

def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # 1️⃣ Read input from form
    budget = int(request.form.get("budget", 0))
    ram = int(request.form.get("ram", 0))
    usage = request.form.get("usage", "")


    # 2️⃣ Apply filtering rules
    filtered_data = data[
        (data["price"] <= budget) &
        (data["ram"] >= ram) &
        (data["usage"] == usage)
    ]

    # 3️⃣ Convert DataFrame to list of dicts
    mobiles = filtered_data.to_dict(orient="records")

    return render_template("results.html", mobiles=mobiles)

if __name__ == "__main__":
    app.run(debug=True)