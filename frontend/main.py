import streamlit as st
import requests
import pandas as pd
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Triage Télémédecine",
    page_icon="🏥",
    layout="wide",
)

st.title("Système de tri d'urgence — Télémédecine")

# =============================================================
# MODÈLE
# =============================================================
# Un seul modèle déployé en production : RandomForest du scénario 2 (éthique)
st.caption("Scénario S2 (éthique, sans sexe ni zone_vie) — modèle : **RandomForest**")

# =============================================================
# FORMULAIRE PATIENT
# =============================================================
st.header("Informations patient")

col1, col2, col3 = st.columns(3)

with col1:
    age            = st.number_input("Âge (ans)",                 min_value=0,    max_value=130,  value=45)
    freq_cardiaque = st.number_input("Fréquence cardiaque (bpm)", min_value=0,    max_value=400,  value=80)
    tension_sys    = st.number_input("Tension systolique (mmHg)", min_value=0,    max_value=400,  value=120)

with col2:
    temp        = st.number_input("Température (°C)",    min_value=20.0, max_value=45.0, value=37.0, step=0.1, format="%.1f")
    sat_oxygene = st.number_input("Saturation O₂ (%)",  min_value=0,    max_value=100,  value=98)
    antecedents = st.selectbox("Antécédents chroniques", options=[0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")

with col3:
    duree_symptomes = st.number_input("Durée des symptômes (h)", min_value=0.0, max_value=2000.0, value=2.0, step=0.5)
    source          = st.selectbox("Source de la demande",        options=["appel", "chat"])

description_symptomes = st.text_area(
    "Description des symptômes",
    placeholder="Décrivez les symptômes du patient...",
    height=120,
)

# =============================================================
# ALERTES CLINIQUES INDÉPENDANTES DU MODÈLE
# =============================================================
# Seuils absolus : ces valeurs sont critiques quelle que soit la prédiction ML
_alertes = []
if sat_oxygene < 80:
    _alertes.append(f"SpO₂ à {sat_oxygene}% — seuil critique : < 80%")
if freq_cardiaque < 30:
    _alertes.append(f"Fréquence cardiaque à {freq_cardiaque} bpm — seuil critique : < 30 bpm")
if freq_cardiaque > 180:
    _alertes.append(f"Fréquence cardiaque à {freq_cardiaque} bpm — seuil critique : > 180 bpm")
if temp < 32.0:
    _alertes.append(f"Température à {temp}°C — seuil critique : < 32°C")
if temp > 41.0:
    _alertes.append(f"Température à {temp}°C — seuil critique : > 41°C")
if tension_sys < 70:
    _alertes.append(f"Tension systolique à {tension_sys} mmHg — seuil critique : < 70 mmHg")
if tension_sys > 200:
    _alertes.append(f"Tension systolique à {tension_sys} mmHg — seuil critique : > 200 mmHg")

if _alertes:
    st.markdown(
        f"""
        <div style="
            background-color: #7b1fa2;
            color: white;
            padding: 16px 20px;
            border-radius: 10px;
            margin: 16px 0;
        ">
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 8px;">
                ⚠️ ALERTE CLINIQUE — Valeurs critiques détectées
            </div>
            <div style="font-size: 13px; opacity: 0.85; margin-bottom: 10px;">
                Indépendant du modèle ML — à traiter en priorité absolue
            </div>
            {"".join(f'<div style="margin: 4px 0;">• {a}</div>' for a in _alertes)}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================
# PRÉDICTION
# =============================================================
if st.button("Prédire le niveau d'urgence", type="primary", use_container_width=True):
    if not description_symptomes.strip():
        st.warning("Veuillez renseigner la description des symptômes.")
    else:
        payload = {
            "age":                   age,
            "freq_cardiaque":        freq_cardiaque,
            "tension_sys":           tension_sys,
            "temp":                  temp,
            "sat_oxygene":           sat_oxygene,
            "antecedents":           antecedents,
            "duree_symptomes":       duree_symptomes,
            "source":                source,
            "description_symptomes": description_symptomes,
        }

        try:
            response = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Impossible de contacter le backend. Vérifiez que FastAPI tourne sur " + BACKEND_URL)
            st.stop()
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur backend : {e.response.json().get('detail', str(e))}")
            st.stop()

        # Couleur selon le niveau d'urgence
        colors = {0: "#2e7d32", 1: "#e65100", 2: "#c62828"}
        label  = result["label"]
        pred   = result["prediction"]
        color  = colors[pred]

        st.markdown(
            f"""
            <div style="
                background-color: {color};
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 28px;
                font-weight: bold;
                margin: 16px 0;
            ">
                {label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Prédit par : **{result['model_name']}** — {result['timestamp']}")

        # Probabilités par classe
        st.subheader("Probabilités par classe")
        probas = result["probabilites"]

        proba_cols = st.columns(3)
        for i, (classe, couleur) in enumerate([
            ("Pas urgent",  "#2e7d32"),
            ("Urgent",      "#e65100"),
            ("Très urgent", "#c62828"),
        ]):
            with proba_cols[i]:
                valeur = probas[classe]
                st.metric(label=classe, value=f"{valeur:.1%}")
                st.progress(valeur)

# =============================================================
# HISTORIQUE DES PRÉDICTIONS
# =============================================================
st.divider()
st.header("Historique des prédictions")

try:
    hist_response = requests.get(f"{BACKEND_URL}/history", timeout=5)
    hist_response.raise_for_status()
    historique = hist_response.json()
except Exception:
    historique = []

if historique:
    df_hist = pd.DataFrame(historique)
    cols_affichees = [
        "timestamp", "age", "freq_cardiaque", "tension_sys", "temp",
        "sat_oxygene", "duree_symptomes", "source", "label",
        "proba_pas_urgent", "proba_urgent", "proba_tres_urgent",
    ]
    df_hist = df_hist[[c for c in cols_affichees if c in df_hist.columns]]
    st.dataframe(df_hist, use_container_width=True)
else:
    st.info("Aucune prédiction enregistrée pour l'instant.")
