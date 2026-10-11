import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.title("Titanic Passenger Survival Prediction System")
st.write("Enter the passenger details and click Predict.")

model = pickle.load(open("svc_trained_model.pkl", "rb"))

pclass_opts = ["First", "Second", "Third"]
gender_opts = ["Male", "Female"]
sibling_opts = ["Zero", "One", "Two", "Three"]
embarked_opts = ["Southampton", "Cherbourg", "Queenstown"]

pclass_enc = LabelEncoder().fit(pclass_opts)
gender_enc = LabelEncoder().fit(gender_opts)
sibling_enc = LabelEncoder().fit(sibling_opts)
embarked_enc = LabelEncoder().fit(embarked_opts)

pclass = st.selectbox("PClass", pclass_opts)
gender = st.selectbox("Gender", gender_opts)
sibling = st.selectbox("Sibling", sibling_opts)
embarked = st.selectbox("Embarked", embarked_opts)

if st.button("Predict"):
    features = pd.DataFrame({
        "PClass": [pclass_enc.transform([pclass])[0]],
        "Gender": [gender_enc.transform([gender])[0]],
        "Sibling": [sibling_enc.transform([sibling])[0]],
        "Embarked": [embarked_enc.transform([embarked])[0]],
    })
    result = model.predict(features)[0]
    if result == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")
