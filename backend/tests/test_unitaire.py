"""
Tests unitaires — logique isolée, sans serveur ni modèles.

On teste uniquement des fonctions ou règles indépendantes :
- Les validateurs Pydantic (bornes physiologiques)
- La discrétisation de l'âge
"""
import pytest
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import PatientInput

# Payload valide de base — on le réutilise dans tous les tests en le modifiant
_BASE = dict(
    age=-9,
    freq_cardiaque=80,
    tension_sys=120,
    temp=37.0,
    sat_oxygene=98,
    antecedents=0,
    duree_symptomes=2.0,
    source="appel",
    description_symptomes="maux de tête légers",
)


# =============================================================
# VALIDATEUR — ÂGE
# =============================================================

def test_age_negatif_rejete():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "age": -1})

def test_age_trop_eleve_rejete():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "age": 131})

def test_age_limite_basse_accepte():
    p = PatientInput(**{**_BASE, "age": 0})
    assert p.age == 0

def test_age_limite_haute_accepte():
    p = PatientInput(**{**_BASE, "age": 130})
    assert p.age == 130


# =============================================================
# VALIDATEUR — SATURATION O₂
# =============================================================

def test_sat_negative_rejetee():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "sat_oxygene": -1})

def test_sat_trop_haute_rejetee():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "sat_oxygene": 101})

def test_sat_zero_acceptee():
    # 0% est physiologiquement critique mais valide comme entrée
    p = PatientInput(**{**_BASE, "sat_oxygene": 0})
    assert p.sat_oxygene == 0

def test_sat_100_acceptee():
    p = PatientInput(**{**_BASE, "sat_oxygene": 100})
    assert p.sat_oxygene == 100


# =============================================================
# VALIDATEUR — TEMPÉRATURE
# =============================================================

def test_temp_trop_basse_rejetee():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "temp": 29.9})

def test_temp_trop_haute_rejetee():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "temp": 45.1})

def test_temp_limite_basse_acceptee():
    p = PatientInput(**{**_BASE, "temp": 30.0})
    assert p.temp == 30.0

def test_temp_limite_haute_acceptee():
    p = PatientInput(**{**_BASE, "temp": 45.0})
    assert p.temp == 45.0


# =============================================================
# VALIDATEUR — ANTÉCÉDENTS
# =============================================================

def test_antecedents_invalide_rejete():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "antecedents": 2})

def test_antecedents_negatif_rejete():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "antecedents": -1})

def test_antecedents_zero_accepte():
    p = PatientInput(**{**_BASE, "antecedents": 0})
    assert p.antecedents == 0

def test_antecedents_un_accepte():
    p = PatientInput(**{**_BASE, "antecedents": 1})
    assert p.antecedents == 1


# =============================================================
# VALIDATEUR — SOURCE
# =============================================================

def test_source_invalide_rejetee():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "source": "sms"})

def test_source_email_rejetee():
    with pytest.raises(ValidationError):
        PatientInput(**{**_BASE, "source": "email"})

def test_source_appel_acceptee():
    p = PatientInput(**{**_BASE, "source": "appel"})
    assert p.source == "appel"

def test_source_chat_acceptee():
    p = PatientInput(**{**_BASE, "source": "chat"})
    assert p.source == "chat"
