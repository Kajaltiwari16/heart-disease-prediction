from flask import Flask, render_template, request
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)

lr_model = pickle.load(open("lr_model.pkl", "rb"))
svm_model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def get_metrics():
    if os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r") as f:
                return json.load(f)
        except:
            pass
    return {'lr_acc': 0, 'svm_acc': 0}

@app.route('/')
def home():
    return render_template('index.html', metrics=get_metrics())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('model')

        data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]

        final_input = scaler.transform([data])

        if model_type == "lr":
            prediction = lr_model.predict(final_input)
            probability = lr_model.predict_proba(final_input)[0]
            model_name = "Logistic Regression"
        else:
            prediction = svm_model.predict(final_input)
            probability = svm_model.predict_proba(final_input)[0]
            model_name = "SVM"

        risk_score = round(probability[1] * 100, 1)
        result = 1 if prediction[0] == 1 else 0

        return render_template('index.html', prediction=result, risk_score=risk_score, model_name=model_name, metrics=get_metrics())

    except Exception as e:
        return render_template('index.html', error=str(e), metrics=get_metrics())

# CSV Upload
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    model_type = request.form['model']

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    if 'output' in df.columns:
        df = df.drop('output', axis=1)

    scaled = scaler.transform(df)

    if model_type == "lr":
        preds = lr_model.predict(scaled)
        probs = lr_model.predict_proba(scaled)[:, 1]
        model_name = "Logistic Regression"
    else:
        preds = svm_model.predict(scaled)
        probs = svm_model.predict_proba(scaled)[:, 1]
        model_name = "SVM"

    df['Prediction (1=Disease, 0=Healthy)'] = preds
    df['Risk Probability (%)'] = [round(p * 100, 1) for p in probs]

    return render_template('index.html',
                           tables=[df.to_html(classes='data', index=False)],
                           batch_result=f"Batch Processed using {model_name}",
                           metrics=get_metrics())

if __name__ == "__main__":
    app.run(debug=True)