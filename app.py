from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('form.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['review']
    if not text.strip():
        return render_template('form.html', prediction="Please enter a review.")
    
    clean_text = text.lower()
    clean_text = clean_text.replace("n't", " not")
    clean_text = clean_text.replace("not ", "not_")

    vec = vectorizer.transform([clean_text])
    pred = model.predict(vec)[0]
    conf = np.max(model.predict_proba(vec)) * 100

    sentiment = "Positive 😊" if pred == 1 else "Negative 😔"
    return render_template(
        'form.html',
        review=text,
        prediction=sentiment,
        confidence=f"{conf:.2f}%"
    )

@app.route('/graphs')
def graphs():
    return render_template('graphs.html')

if __name__ == "__main__":
    app.run(debug=True)
