"""
Tests fonctionnels — API testée de bout en bout via HTTP.

On démarre le serveur avec TestClient (FastAPI) et on envoie
de vraies requêtes HTTP pour vérifier le comportement global.

Trois axes prioritaires :
1. Contrat de l'API (structure de la réponse)
2. Gestion des erreurs (codes HTTP corrects)
3. Contrainte médicale (un patient critique n'est jamais "Pas urgent")
"""
import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app

client = TestClient(app)

# Payload valide de base
_BASE = dict(
    age=45,
    freq_cardiaque=80,
    tension_sys=120,
    temp=37.0,
    sat_oxygene=98,
    antecedents=0,
    duree_symptomes=2.0,
    source="appel",
    description_symptomes="maux de tête légers",
    model_name="RandomForest",
)

# Patient avec constantes critiques — valeurs proches des moyennes classe 2 du dataset
# FC~120, temp~39°C, SpO₂~88%, tension~175 → on prend des valeurs encore plus extrêmes
_PATIENT_CRITIQUE = dict(
    age=55,
    freq_cardiaque=145,
    tension_sys=185,
    temp=40.2,
    sat_oxygene=79,
    antecedents=1,
    duree_symptomes=1.0,
    source="appel",
    description_symptomes="détresse respiratoire sévère, douleur thoracique, confusion",
    model_name="RandomForest",
)


# =============================================================
# ENDPOINTS DE BASE
# =============================================================

def test_health_retourne_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_contient_status_ok():
    response = client.get("/health")
    assert response.json()["status"] == "ok"

def test_models_retourne_200():
    response = client.get("/models")
    assert response.status_code == 200

def test_models_liste_non_vide():
    response = client.get("/models")
    assert len(response.json()["models"]) > 0

def test_models_contient_random_forest():
    response = client.get("/models")
    assert "RandomForest" in response.json()["models"]


# =============================================================
# CONTRAT DE L'API — STRUCTURE DE LA RÉPONSE
# =============================================================

def test_predict_retourne_200():
    response = client.post("/predict", json=_BASE)
    assert response.status_code == 200

def test_predict_contient_toutes_les_cles():
    result = client.post("/predict", json=_BASE).json()
    assert "prediction"   in result
    assert "label"        in result
    assert "probabilites" in result
    assert "model_name"   in result
    assert "timestamp"    in result

def test_predict_probabilites_trois_classes():
    result = client.post("/predict", json=_BASE).json()
    assert set(result["probabilites"].keys()) == {"Pas urgent", "Urgent", "Très urgent"}

def test_predict_probabilites_somment_a_1():
    # Invariant mathématique : les probas doivent toujours sommer à 1.0
    result = client.post("/predict", json=_BASE).json()
    total = sum(result["probabilites"].values())
    assert abs(total - 1.0) < 0.01

def test_predict_label_coherent_avec_prediction():
    # Le label textuel doit correspondre à la classe numérique
    labels = {0: "Pas urgent", 1: "Urgent", 2: "Très urgent"}
    result = client.post("/predict", json=_BASE).json()
    assert result["label"] == labels[result["prediction"]]

def test_predict_model_name_retourne_dans_reponse():
    result = client.post("/predict", json=_BASE).json()
    assert result["model_name"] == "RandomForest"


# =============================================================
# GESTION DES ERREURS
# =============================================================

def test_predict_modele_inexistant_retourne_400():
    payload = {**_BASE, "model_name": "ModeleInvente"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

def test_predict_age_invalide_retourne_422():
    payload = {**_BASE, "age": -1}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_sat_invalide_retourne_422():
    payload = {**_BASE, "sat_oxygene": 150}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_temp_invalide_retourne_422():
    payload = {**_BASE, "temp": 20.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_source_invalide_retourne_422():
    payload = {**_BASE, "source": "email"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# =============================================================
# CONTRAINTE MÉDICALE — TEST ÉTHIQUE PRIORITAIRE
# =============================================================

def test_patient_critique_jamais_pas_urgent():
    """
    Contrainte éthique fondamentale : un patient avec des constantes vitales
    clairement critiques (FC élevée, SpO₂ basse, fièvre, hypertension) ne doit
    JAMAIS être prédit "Pas urgent" (classe 0).

    Ce test doit exploser immédiatement si quelqu'un modifie les seuils,
    le class_weight ou la logique de prédiction de façon dangereuse.
    """
    result = client.post("/predict", json=_PATIENT_CRITIQUE).json()
    assert result["prediction"] != 0, (
        f"ALERTE ÉTHIQUE : un patient critique a été classé 'Pas urgent'. "
        f"Probabilités : {result['probabilites']}"
    )

def test_patient_critique_probabilite_vitale_non_nulle():
    """
    La probabilité de classe 2 pour un patient critique ne doit jamais être 0.
    Si elle est 0, le modèle n'a pas du tout considéré l'urgence vitale.
    """
    result = client.post("/predict", json=_PATIENT_CRITIQUE).json()
    assert result["probabilites"]["Très urgent"] > 0.0, (
        "La probabilité vitale est nulle pour un patient en état critique."
    )
