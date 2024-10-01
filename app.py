import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(f"Received data: {data}")
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(new_data)
        print(f"Model output: {output[0]}")
        return jsonify({"prediction": output[0]})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The house price prediction is {output}")

if __name__ == "__main__":
    app.run(debug=True)
