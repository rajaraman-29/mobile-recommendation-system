from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data/data.csv")

# Encode usage
usage_encoder = LabelEncoder()
data["usage_encoded"] = usage_encoder.fit_transform(data["usage"])

# Features for similarity
feature_cols = ["price", "ram", "storage", "rating", "usage_encoded"]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[feature_cols])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    budget = int(request.form.get("budget"))
    ram = int(request.form.get("ram"))
    usage = request.form.get("usage")

    usage_encoded = usage_encoder.transform([usage])[0]

    user_vector = [[
        budget,
        ram,
        data["storage"].mean(),
        data["rating"].mean(),
        usage_encoded
    ]]

    user_vector_scaled = scaler.transform(user_vector)

    similarity_scores = cosine_similarity(
        user_vector_scaled,
        scaled_features
    )[0]

    data["similarity"] = similarity_scores

    filtered = data[
        (data["price"] <= budget) &
        (data["ram"] >= ram)
    ].sort_values(by="similarity", ascending=False)

    mobiles = filtered.head(5).to_dict(orient="records")

    # Add best price + reason
    for mobile in mobiles:
        store_prices = {
            "Amazon": mobile["amazon_price"],
            "Flipkart": mobile["flipkart_price"],
            "Croma": mobile["croma_price"]
        }

        best_store = min(store_prices, key=store_prices.get)
        best_price = store_prices[best_store]

        mobile["best_store"] = best_store
        mobile["best_price"] = best_price
        mobile["reason"] = (
            f"Within your budget, has {mobile['ram']}GB RAM "
            f"and suitable for {usage} usage."
        )

    return render_template("results.html", mobiles=mobiles)

if __name__ == "__main__":
    app.run(debug=True)
