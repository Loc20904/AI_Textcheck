from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model và vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/moderate', methods=['POST'])
def moderate():
    data = request.get_json()
    text = data.get('text', '')

    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    return jsonify({
        'text': text,
        'sensitive': bool(pred),
        'confidence': proba
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
