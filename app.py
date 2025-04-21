import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Charger le modèle et le scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.gz")

# Titre de l'application
st.title("🌍 Prédiction de la Qualité de l'Air - CO(GT)")

st.markdown("""
Entrez les 24 dernières valeurs horaires de CO(GT) (concentration du monoxyde de carbone) 
pour obtenir la prédiction de la prochaine heure.
""")

# Interface utilisateur : saisir les 24 valeurs
input_values = []
for i in range(24):
    val = st.number_input(f"CO(GT) - Heure {i+1}", min_value=0.0, max_value=50.0, step=0.1, key=i)
    input_values.append(val)

# Prédiction
if st.button("🔮 Prédire la prochaine valeur"):
    seq = np.array(input_values).reshape(-1, 1)
    seq_scaled = scaler.transform(seq)
    seq_scaled = seq_scaled.reshape(1, 24, 1)
    prediction_scaled = model.predict(seq_scaled)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    st.success(f"✅ Prévision de CO(GT) pour l'heure suivante : **{prediction:.2f}**")

