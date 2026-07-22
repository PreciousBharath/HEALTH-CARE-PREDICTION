from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load model and encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('le_sex.pkl', 'rb') as f:
    le_sex = pickle.load(f)

with open('le_smoker.pkl', 'rb') as f:
    le_smoker = pickle.load(f)

with open('le_region.pkl', 'rb') as f:
    le_region = pickle.load(f)

# Load dataset for graphs
df = pd.read_csv('insurance.csv')

# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Encode categorical values
        sex_encoded = le_sex.transform([sex])[0]
        smoker_encoded = le_smoker.transform([smoker])[0]
        region_encoded = le_region.transform([region])[0]

        # Prepare input
        X_input = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

        # Prediction
        predicted_charge = model.predict(X_input)[0]

        # ---------- Create Graphs ----------
        # Age vs Avg Charges
        age_charges = df.groupby("age")["charges"].mean()
        plt.figure(figsize=(6,4))
        plt.plot(age_charges.index, age_charges.values, marker='o', color='orange')
        plt.xlabel("Age")
        plt.ylabel("Average Charges")
        plt.title("Average Charges by Age")
        age_graph = "static/age_charges.png"
        plt.savefig(age_graph)
        plt.close()

        # Smoker vs Charges
        plt.figure(figsize=(6,4))
        sns.barplot(x="smoker", y="charges", data=df)
        plt.title("Charges: Smoker vs Non-Smoker")
        smoker_graph = "static/smoker_charges.png"
        plt.savefig(smoker_graph)
        plt.close()

        return render_template('result.html', 
                               predicted_charge=round(predicted_charge, 2),
                               age_graph=age_graph,
                               smoker_graph=smoker_graph)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    # Ensure static folder exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
