# -------------------- IMPORTS --------------------
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- APP INIT --------------------
app = Flask(__name__)

# -------------------- LOAD DATA --------------------
data = pd.read_csv("data/data.csv")

# Encode usage (student / gaming / office)
usage_encoder = LabelEncoder()
data["usage_encoded"] = usage_encoder.fit_transform(data["usage"])

# Features used for recommendation
feature_columns = ["price", "ram", "storage", "rating", "usage_encoded"]

# Scale features for similarity calculation
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[feature_columns])

# -------------------- ROUTES --------------------

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# -------------------- RECOMMENDATION (POST ONLY) --------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    # Get user inputs safely
    budget = int(request.form.get("budget", 0))
    ram = int(request.form.get("ram", 0))
    usage = request.form.get("usage", "student")

    # Encode usage
    usage_encoded = usage_encoder.transform([usage])[0]

    # Create user preference vector
    user_vector = [[
        budget,
        ram,
        data["storage"].mean(),
        data["rating"].mean(),
        usage_encoded
    ]]

    # Scale user vector
    user_vector_scaled = scaler.transform(user_vector)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector_scaled,
        scaled_features
    )[0]

    # Attach similarity to dataset
    data["similarity"] = similarity_scores

    # Filter mobiles based on constraints
    filtered = data[
        (data["price"] <= budget) &
        (data["ram"] >= ram)
    ].sort_values(by="similarity", ascending=False)

    # Take top 5 results
    mobiles = filtered.head(5).to_dict(orient="records")

    # Add price comparison & explanation
    for mobile in mobiles:
        prices = {
            "Amazon": mobile["amazon_price"],
            "Flipkart": mobile["flipkart_price"],
            "Croma": mobile["croma_price"]
        }

        best_store = min(prices, key=prices.get)
        mobile["best_store"] = best_store
        mobile["best_price"] = prices[best_store]

        mobile["reason"] = (
            f"Fits your budget, has {mobile['ram']}GB RAM "
            f"and suitable for {usage} usage."
        )

    # IMPORTANT: Render results page
    return render_template("results.html", mobiles=mobiles)

# -------------------- SAFETY REDIRECT (GET /recommend) --------------------
@app.route("/recommend", methods=["GET"])
def recommend_get():
    # Prevent direct access to /recommend
    return redirect(url_for("home"))

# -------------------- COMPARISON + FINAL RECOMMENDATION --------------------
@app.route("/compare", methods=["POST"])
def compare():
    selected_models = request.form.getlist("compare")

    # Must select exactly two mobiles
    if len(selected_models) != 2:
        return "<h3>Please select exactly 2 mobiles.</h3><a href='/'>Go Back</a>"

    compare_data = data[data["model"].isin(selected_models)]
    mobiles = compare_data.to_dict(orient="records")

    # Compute best price and comparison score
    for mobile in mobiles:
        prices = {
            "Amazon": mobile["amazon_price"],
            "Flipkart": mobile["flipkart_price"],
            "Croma": mobile["croma_price"]
        }

        mobile["best_store"] = min(prices, key=prices.get)
        mobile["best_price"] = prices[mobile["best_store"]]

        # Simple scoring logic
        mobile["score"] = (
            mobile["ram"] * 2 +
            mobile["rating"] * 3 -
            (mobile["best_price"] / 10000)
        )

    # Decide final recommendation
    recommended_mobile = max(mobiles, key=lambda x: x["score"])

    return render_template(
        "compare.html",
        mobiles=mobiles,
        recommended=recommended_mobile
    )

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app.run(debug=True)
