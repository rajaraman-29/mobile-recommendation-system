from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

data = pd.read_csv("data/data.csv")

usage_encoder = LabelEncoder()
data["usage_encoded"] = usage_encoder.fit_transform(data["usage"])

features = ["price", "ram", "storage", "rating", "usage_encoded"]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[features])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    budget = int(request.form["budget"])
    ram = int(request.form["ram"])
    usage = request.form["usage"]

    usage_encoded = usage_encoder.transform([usage])[0]

    user_vector = [[budget, ram, data["storage"].mean(), data["rating"].mean(), usage_encoded]]
    user_vector_scaled = scaler.transform(user_vector)

    similarity = cosine_similarity(user_vector_scaled, scaled_features)[0]
    data["similarity"] = similarity

    filtered = data[(data["price"] <= budget) & (data["ram"] >= ram)]
    filtered = filtered.sort_values(by="similarity", ascending=False)

    mobiles = filtered.head(5).to_dict(orient="records")

    for m in mobiles:
        prices = {
            "Amazon": m["amazon_price"],
            "Flipkart": m["flipkart_price"],
            "Croma": m["croma_price"]
        }

        m["best_store"] = min(prices, key=prices.get)
        m["best_price"] = prices[m["best_store"]]

        # ✅ EXPLICITLY PASS LINKS
        m["amazon_link"] = m["amazon_link"]
        m["flipkart_link"] = m["flipkart_link"]
        m["croma_link"] = m["croma_link"]

        reasons = []
        if m["ram"] >= ram:
            reasons.append("sufficient RAM")
        if m["rating"] >= 4:
            reasons.append("high rating")
        if m["best_price"] <= budget:
            reasons.append("fits your budget")

        m["reason"] = ", ".join(reasons)

    return render_template("results.html", mobiles=mobiles)

@app.route("/recommend", methods=["GET"])
def recommend_get():
    return redirect(url_for("home"))

@app.route("/compare", methods=["POST"])
def compare():
    selected = request.form.getlist("compare")

    mobiles = data[data["model"].isin(selected)].to_dict(orient="records")

    for m in mobiles:
        prices = {
            "Amazon": m["amazon_price"],
            "Flipkart": m["flipkart_price"],
            "Croma": m["croma_price"]
        }
        m["best_store"] = min(prices, key=prices.get)
        m["best_price"] = prices[m["best_store"]]
        m["score"] = m["ram"] * 2 + m["rating"] * 3 - (m["best_price"] / 10000)

    winner = max(mobiles, key=lambda x: x["score"])
    winner["explanation"] = "Higher RAM, better rating, and lower price."

    return render_template("compare.html", mobiles=mobiles, recommended=winner)

if __name__ == "__main__":
    app.run(debug=True)
