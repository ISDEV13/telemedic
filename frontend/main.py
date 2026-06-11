import streamlit as st
import requests

st.title("Calculateur de carré")

nombre = st.number_input("Entrez un nombre", value=0.0)

if st.button("Calculer"):
    response = requests.post("http://backend:8000/carre/", json={"nombre": nombre})
    if response.status_code == 200:
        data = response.json()
        st.success(f"Le carré de {data['nombre']} est {data['carre']}")
    else:
        st.error("Erreur lors de l'appel à l'API")
