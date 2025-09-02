from flask import Flask, render_template, request
import pickle 

with open("fake_news_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)



@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        return render_template("prediction.html", username=username, email=email)
    return render_template("login.html")  

@app.route("/predict",methods = ["GET","POST"])
def predict():
     if request.method == "POST":
        news_text = request.form["news"] 
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        label = "Real News" if prediction == 1 else "Fake News"
        return render_template("thankyou.html", prediction=label)
    
if __name__ == "__main__":
    app.run(debug=True)
