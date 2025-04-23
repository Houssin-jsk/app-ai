from flask import Flask, render_template, request
import pickle
import pandas as pd

# Charger le modèle
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger l'ordre exact des features
with open('feature_order.pkl', 'rb') as f:
    feature_order = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            credit_score = float(request.form['CreditScore'])
            geography = request.form['Geography']
            gender = request.form['Gender']
            age = float(request.form['Age'])
            tenure = float(request.form['Tenure'])
            balance = float(request.form['Balance'])
            num_products = float(request.form['NumOfProducts'])
            has_cr_card = int(request.form['HasCrCard'])
            is_active = int(request.form['IsActiveMember'])
            salary = float(request.form['EstimatedSalary'])

            # Encodage
            geo_germany = 1 if geography == 'Germany' else 0
            geo_spain = 1 if geography == 'Spain' else 0
            gender_encoded = 1 if gender == 'Male' else 0

            # Créer dictionnaire brut
            input_data = {
                'CreditScore': credit_score,
                'Geography_Germany': geo_germany,
                'Geography_Spain': geo_spain,
                'Gender': gender_encoded,
                'Age': age,
                'Tenure': tenure,
                'Balance': balance,
                'NumOfProducts': num_products,
                'HasCrCard': has_cr_card,
                'IsActiveMember': is_active,
                'EstimatedSalary': salary
            }

            # Créer DataFrame et réordonner
            df_input = pd.DataFrame([input_data])
            df_input = df_input[feature_order]  # 🔁 Match exact order

            result = model.predict(df_input)[0]
            prediction = f"Prediction: {result}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
