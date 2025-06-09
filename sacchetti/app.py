from flask import Flask, request, jsonify
import pandas as pd
import joblib
app = Flask(__name__)
model = joblib.load('sacchetti/bestHeartPred.joblib')

@app.route('/infer', methods=['POST'])
def hello():
    data = request.get_json()
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        prediction = model.predict(input_df)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)