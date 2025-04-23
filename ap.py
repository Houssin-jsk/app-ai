import streamlit as st
import pandas as pd
import pickle
import os
import pickle

# تحديد المسار ديال الملفات بناءً على موقع السكريبت
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "random_forest_model.pkl")
feature_order_path = os.path.join(base_path, "feature_order.pkl")

# تحميل الموديل
with open(model_path, "rb") as file:
    model = pickle.load(file)

# تحميل ترتيب الأعمدة
with open(feature_order_path, 'rb') as f:
    feature_order = pickle.load(f)


# Titre de l'application
st.title("Churn Prediction App")
st.write("This app predicts customer churn based on various features.")

# Collecte des données utilisateur
credit_score = float(st.number_input("Credit Score", min_value=0.0, max_value=1000.0))
geography = st.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.selectbox("Gender", ("Male", "Female"))
age = st.number_input("Age", min_value=0, max_value=120)
tenure = st.number_input("Tenure", min_value=0, max_value=20)
balance = st.number_input("Balance", min_value=0.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=10)
has_cr_card_text = st.selectbox("Has Credit Card", ("Yes", "No"))
is_active_text = st.selectbox("Is Active Member", ("Yes", "No"))
salary = st.number_input("Estimated Salary", min_value=0.0)

# Barre latérale pour l'historique
st.sidebar.title("Prediction History")
if "history" not in st.session_state:
    st.session_state.history = []

# Encodage des variables catégorielles
geo_germany = 1 if geography == 'Germany' else 0
geo_spain = 1 if geography == 'Spain' else 0
gender_encoded = 1 if gender == 'Male' else 0
has_cr_card = 1 if has_cr_card_text == "Yes" else 0
is_active = 1 if is_active_text == "Yes" else 0

# Bouton de prédiction
if st.button("Predict"):
    try:
        # Créer un DataFrame d'entrée avec les colonnes dans le bon ordre
        df_input = pd.DataFrame([[credit_score, geo_germany, geo_spain, gender_encoded, age, tenure, balance,
                                  num_products, has_cr_card, is_active, salary]],
                                columns=feature_order)

        # Faire la prédiction
        result = model.predict(df_input)[0]

        # Affichage du résultat
        if result == 1:
            st.success("The customer is **not likely** to churn.")
        else:
            st.info("The customer is **likely** to churn.")

        # Historique dans la sidebar
        summary = f"""
        ----------------------------------------
        **Input:**  
        Credit Score: {credit_score}  
        Geography: {geography}  
        Gender: {gender}  
        Age: {age}  
        Tenure: {tenure}  
        Balance: {balance}  
        Products: {num_products}  
        Credit Card: {has_cr_card_text}  
        Active: {is_active_text}  
        Salary: {salary}  

        **Prediction:** {"The customer is **likely** to churn." if result == 1 else "The customer is **not likely** to churn."}
        ----------------------------------------
        """
        st.session_state.history.append(summary)

    except Exception as e:
        st.error(f"Error: {e}")

# Affichage de l'historique
for i, entry in enumerate(st.session_state.history[::-1]):
    st.sidebar.markdown(f"### Prediction {len(st.session_state.history)-i}")
    st.sidebar.markdown(entry)
