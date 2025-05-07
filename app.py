from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('aqi_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        co = float(request.form['co'])
        ozone = float(request.form['ozone'])
        no2 = float(request.form['no2'])
        pm25 = float(request.form['pm25'])

        features = np.array([[co, ozone, no2, pm25]])
        prediction = model.predict(features)[0]
        return render_template('index.html', prediction_text=f'Predicted AQI Value: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

