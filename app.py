from flask import Flask,render_template,request

app= Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/recommend",methods=["POST"])
def recommend():
    budget=request.form["budget"]
    ram=request.form["ram"]
    usage=request.form["usage"]

    return f"""
     <h2>Preferences Received</h2>
    <p>Budget: ₹{budget}</p>
    <p>RAM: {ram} GB</p>
    <p>Usage: {usage}</p>
    """
if __name__=="__main__":
    app.run(debug=True)
