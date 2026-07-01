import os
import csv
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Télémédecine — API de triage")
Instrumentator().instrument(app).expose(app)

# =============================================================
# CHARGEMENT DU PIPELINE (préprocesseur + tfidf partagés)
# =============================================================
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Modèle unique déployé en production : RandomForest du scénario 2 (éthique)
_MODEL_NAME = "RandomForest"

try:
    preprocessor     = joblib.load(os.path.join(_MODELS_DIR, "S2_preprocessor.pkl"))
    tfidf            = joblib.load(os.path.join(_MODELS_DIR, "S2_tfidf.pkl"))
    model            = joblib.load(os.path.join(_MODELS_DIR, "S2_RandomForest.pkl"))
    _PIPELINE_LOADED = True
except FileNotFoundError:
    preprocessor     = None
    tfidf            = None
    model            = None
    _PIPELINE_LOADED = False

# =============================================================
# CONSTANTES DE PÉNALISATION (identiques à l'entraînement)
# =============================================================
_THRESHOLDS = {2: 0.20, 1: 0.30}
_AGE_BINS   = [0, 17, 40, 64, float("inf")]
_AGE_LABELS = ["enfant", "adulte_jeune", "adulte", "senior"]
_NUM_ETH    = ["freq_cardiaque", "tension_sys", "temp", "sat_oxygene", "antecedents", "duree_symptomes"]
_CAT_ETH    = ["source", "age"]
_LABELS     = {0: "Pas urgent", 1: "Urgent", 2: "Très urgent"}

# =============================================================
# HISTORIQUE DES PRÉDICTIONS
# =============================================================
_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "predictions_history.csv")
_HISTORY_COLS = [
    "timestamp", "age", "freq_cardiaque", "tension_sys", "temp",
    "sat_oxygene", "antecedents", "duree_symptomes", "source",
    "description_symptomes", "prediction", "label",
    "proba_pas_urgent", "proba_urgent", "proba_tres_urgent",
]

def _init_history():
    if not os.path.exists(_HISTORY_PATH):
        with open(_HISTORY_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_HISTORY_COLS).writeheader()

def _append_history(row: dict):
    with open(_HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=_HISTORY_COLS).writerow(row)

_init_history()

# =============================================================
# SCHÉMA D'ENTRÉE
# =============================================================
class PatientInput(BaseModel):
    age:                   float
    freq_cardiaque:        float
    tension_sys:           float
    temp:                  float
    sat_oxygene:           float
    antecedents:           int
    duree_symptomes:       float
    source:                str
    description_symptomes: str

    @field_validator("age")
    @classmethod
    def age_valide(cls, v):
        if v < 0 or v > 130:
            raise ValueError("L'âge doit être compris entre 0 et 130 ans")
        return v

    @field_validator("sat_oxygene")
    @classmethod
    def sat_valide(cls, v):
        if v < 0 or v > 100:
            raise ValueError("La saturation doit être comprise entre 0 et 100 %")
        return v

    @field_validator("temp")
    @classmethod
    def temp_valide(cls, v):
        if v < 30 or v > 45:
            raise ValueError("La température doit être comprise entre 30 et 45 °C")
        return v

    @field_validator("antecedents")
    @classmethod
    def antecedents_valide(cls, v):
        if v not in (0, 1):
            raise ValueError("antecedents doit valoir 0 ou 1")
        return v

    @field_validator("source")
    @classmethod
    def source_valide(cls, v):
        if v not in ("appel", "chat"):
            raise ValueError("source doit être 'appel' ou 'chat'")
        return v

# =============================================================
# ENDPOINTS
# =============================================================

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "pipeline_loaded": _PIPELINE_LOADED,
        "model":           _MODEL_NAME,
    }


@app.post("/predict")
def predict(patient: PatientInput):
    if not _PIPELINE_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Pipeline non chargé — lancez d'abord le script d'entraînement pour générer les fichiers pkl.",
        )

    # 1. Construire un DataFrame avec les champs bruts
    df = pd.DataFrame([patient.model_dump()])

    # 2. Discrétiser l'âge (identique à l'entraînement S2)
    df["age"] = pd.cut(
        df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=True
    )

    # 3. Extraire les colonnes tabulaires dans l'ordre attendu par le préprocesseur
    tab = df[_NUM_ETH + _CAT_ETH]

    # 4. Appliquer le préprocesseur (imputation + scaling + OHE)
    X_tab = preprocessor.transform(tab)

    # 5. Vectoriser la description texte avec le TF-IDF entraîné
    X_txt = tfidf.transform(df["description_symptomes"].fillna(""))

    # 6. Fusionner les features tabulaires et textuelles
    X = sp.hstack([sp.csr_matrix(X_tab), X_txt], format="csr")

    # 7. Prédire avec les seuils abaissés (mêmes que l'entraînement)
    proba   = model.predict_proba(X)[0]
    classes = model.classes_
    idx     = {cls: i for i, cls in enumerate(classes)}

    if proba[idx[2]] >= _THRESHOLDS[2]:
        prediction = 2
    elif proba[idx[1]] >= _THRESHOLDS[1]:
        prediction = 1
    else:
        prediction = 0

    probas_dict = {
        "Pas urgent":  round(float(proba[idx[0]]), 4),
        "Urgent":      round(float(proba[idx[1]]), 4),
        "Très urgent": round(float(proba[idx[2]]), 4),
    }

    # 8. Journaliser la prédiction dans le CSV
    _append_history({
        "timestamp":             datetime.now().isoformat(timespec="seconds"),
        "age":                   patient.age,
        "freq_cardiaque":        patient.freq_cardiaque,
        "tension_sys":           patient.tension_sys,
        "temp":                  patient.temp,
        "sat_oxygene":           patient.sat_oxygene,
        "antecedents":           patient.antecedents,
        "duree_symptomes":       patient.duree_symptomes,
        "source":                patient.source,
        "description_symptomes": patient.description_symptomes,
        "prediction":            prediction,
        "label":                 _LABELS[prediction],
        "proba_pas_urgent":      probas_dict["Pas urgent"],
        "proba_urgent":          probas_dict["Urgent"],
        "proba_tres_urgent":     probas_dict["Très urgent"],
    })

    return {
        "prediction":   prediction,
        "label":        _LABELS[prediction],
        "probabilites": probas_dict,
        "model_name":   _MODEL_NAME,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/history")
def history():
    if not os.path.exists(_HISTORY_PATH):
        return []
    df = pd.read_csv(_HISTORY_PATH)
    return df.to_dict(orient="records")
