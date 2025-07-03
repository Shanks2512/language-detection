# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_text = request.form['text']
        if not user_text.strip():
            return render_template('index.html', prediction="Please enter some text.", user_text="")
        data = vectorizer.transform([user_text]).toarray()
        prediction = model.predict(data)
        return render_template('index.html', prediction=prediction[0], user_text=user_text)
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == '__main__':
    app.run(debug=True)
