from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data/data.csv")

# Encode categorical column: usage
usage_encoder = LabelEncoder()
data["usage_encoded"] = usage_encoder.fit_transform(data["usage"])

# Select features for ML
feature_columns = ["price", "ram", "storage", "rating", "usage_encoded"]

# Scale numeric features to same range
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[feature_columns])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    # 1️⃣ Get user input
    budget = int(request.form.get("budget", 0))
    ram = int(request.form.get("ram", 0))
    usage = request.form.get("usage", "student")

    # 2️⃣ Encode usage
    usage_encoded = usage_encoder.transform([usage])[0]

    # 3️⃣ Create user preference vector
    user_vector = [[
        budget,
        ram,
        data["storage"].mean(),
        data["rating"].mean(),
        usage_encoded
    ]]

    # 4️⃣ Scale user vector
    user_vector_scaled = scaler.transform(user_vector)

    # 5️⃣ Compute cosine similarity
    similarity_scores = cosine_similarity(
        user_vector_scaled,
        scaled_features
    )[0]

    # 6️⃣ Add similarity score to dataframe
    data["similarity"] = similarity_scores

    # 7️⃣ Filter by basic constraints
    filtered_data = data[
        (data["price"] <= budget) &
        (data["ram"] >= ram)
    ]

    # 8️⃣ Sort by similarity (best first)
    filtered_data = filtered_data.sort_values(
        by="similarity",
        ascending=False
    )

    # 9️⃣ Take top 5 recommendations
    top_mobiles = filtered_data.head(5)

    mobiles = top_mobiles.to_dict(orient="records")

    # Add explanation for each recommendation
    for mobile in mobiles:
        reasons = []

        if mobile["price"] <= budget:
            reasons.append("fits within your budget")

        if mobile["ram"] >= ram:
            reasons.append(f"meets your RAM requirement ({mobile['ram']} GB)")

        if mobile["usage"] == usage:
            reasons.append("suitable for your usage")

        if mobile["rating"] >= 4.5:
            reasons.append("high user rating")

        mobile["reason"] = ", ".join(reasons)


    return render_template("results.html", mobiles=mobiles)

if __name__ == "__main__":
    app.run(debug=True)
