# =============================================================================
# Scénario 1 – Approche multimodale complète
#
# Ce scénario utilise TOUTES les données disponibles :
#   - les colonnes tabulaires (âge, fréquence cardiaque, tension, etc.)
#   - la description textuelle des symptômes (convertie en vecteurs TF-IDF)
#
# Il tourne DEUX FOIS, avec deux configurations différentes :
#   1. AVEC pénalisation : on pénalise le modèle quand il rate les urgences vitales
#   2. SANS pénalisation : comportement "standard" du modèle, pour comparer
#
# Les résultats (graphiques, matrices) sont sauvegardés dans :
#   artifacts/avec_penalisation/
#   artifacts/sans_penalisation/
# =============================================================================

import sys
import os
import time
import warnings
import numpy as np
import scipy.sparse as sp   # pour manipuler les matrices creuses (TF-IDF)
import matplotlib
matplotlib.use("Agg")  # évite d'ouvrir une fenêtre graphique → génère les PNG en arrière-plan
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer  # convertit le texte en vecteurs numériques
from sklearn.neighbors import KNeighborsClassifier           # KNN : classe selon les k voisins les plus proches
from sklearn.linear_model import LogisticRegression          # régression logistique : modèle linéaire de base
from sklearn.ensemble import RandomForestClassifier          # forêt de décision : ensemble d'arbres
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from xgboost import XGBClassifier    # gradient boosting optimisé (arbre + correction itérative)
from lightgbm import LGBMClassifier  # variante de boosting, plus rapide sur grandes données
from tensorflow import keras                           # réseau de neurones
from tensorflow.keras.callbacks import EarlyStopping  # arrête l'entraînement quand ça ne progresse plus

# Remonte d'un niveau (backend/) pour pouvoir importer les modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modules.preprocess import (
    preprocessingTechnique,  # pipeline sklearn : imputation + normalisation + encodage
    apply_business_rules,    # filtrage des lignes aberrantes selon les règles métier
    _DATA_PATH,  # chemin vers dataset_telemed.csv
    _TRASH,      # colonnes à supprimer (identifiants, dates inutiles)
    _NUM_COL,    # colonnes numériques (âge, tension, etc.)
    _CAT_COL,    # colonnes catégorielles (sexe, zone_vie, source)
    _TXT_COL,    # colonne texte libre (description_symptomes) — traitée via TF-IDF
)
from modules.evaluate import (
    evaluate_model,          # calcule et sauvegarde métriques + matrice de confusion
    profile_model,           # mesure temps d'inférence et consommation RAM/CPU
    benchmark_models,        # tableau comparatif des métriques de tous les modèles
    benchmark_metrics_chart, # graphiques en barres comparant F1 / recall / précision
    benchmark_resources,     # tableau comparatif des temps et ressources
    benchmark_confusion,     # toutes les matrices de confusion côte à côte
)
import pandas as pd


# =============================================================================
# PARAMÈTRES GLOBAUX
# Définis ici une seule fois — réutilisés dans les deux configurations.
# =============================================================================

TEXT_COL     = _TXT_COL[0]  # colonne texte à vectoriser avec TF-IDF (source unique : preprocess.py)
TARGET       = "niveau_urgence"          # variable cible (0=pas urgent, 1=urgent, 2=vital)
N_FOLDS      = 5                         # validation croisée en 5 blocs
RANDOM_STATE = 42                        # graine aléatoire → résultats reproductibles

# Noms lisibles affichés dans les graphiques et matrices de confusion
LABEL_NAMES = {0: "Pas urgent", 1: "Urgent", 2: "Très urgent"}
CLASSES     = [0, 1, 2]

# Dossier parent des artifacts — chaque configuration aura son propre sous-dossier
ARTIFACTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")


# =============================================================================
# CONFIGURATIONS À COMPARER
#
# On exécute le même pipeline deux fois avec des réglages différents.
# Cela permet de mesurer l'impact concret des leviers éthiques sur les métriques.
# =============================================================================

CONFIGURATIONS = [
    {
        # ── Configuration 1 : AVEC pénalisation ──────────────────────────────
        # Levier 1 — class_weight : pendant l'entraînement, les erreurs sur la
        #   classe vitale (2) coûtent 10× plus cher → le modèle les évite davantage
        # Levier 2 — thresholds : pendant la prédiction, on abaisse le seuil à 0.25
        #   pour la classe vitale → on prédit "vital" dès que la probabilité dépasse 25 %
        #   (au lieu de 33 % avec l'argmax classique sur 3 classes)
        "nom":          "avec_penalisation",
        "class_weight": {0: 1, 1: 3, 2: 10},  # ratio de pénalisation : vital = 10× pas urgent
        "thresholds":   {2: 0.25, 1: 0.35},    # seuils abaissés pour capturer plus d'urgences
    },
    {
        # ── Configuration 2 : SANS pénalisation (référence) ──────────────────
        # Le modèle est entraîné normalement, sans favoriser aucune classe.
        # On utilise l'argmax classique : on prédit la classe avec la probabilité la plus haute.
        # Sert de base de comparaison pour mesurer l'apport de la pénalisation.
        "nom":          "sans_penalisation",
        "class_weight": None,   # pas de pondération → toutes les erreurs coûtent pareil
        "thresholds":   None,   # argmax classique → prédit la classe la plus probable
    },
]


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def predict_fn(proba, model_classes, thresholds):
    """
    Aiguilleur de prédiction : avec ou sans seuils personnalisés.

    Sans thresholds → argmax : on prend simplement la classe avec la proba la plus haute.
    Avec thresholds → on applique des seuils asymétriques pour favoriser le recall vital.

    proba         : tableau (n_échantillons, n_classes) de probabilités sorties du modèle
    model_classes : ordre des classes dans predict_proba (peut ne pas être [0, 1, 2])
    thresholds    : dict {2: 0.25, 1: 0.35} ou None
    """
    if thresholds is not None:
        return predict_with_thresholds(proba, model_classes, thresholds)

    # Argmax classique : index de la proba maximale → traduit en vrai label de classe
    idx_to_cls = {i: cls for i, cls in enumerate(model_classes)}
    return np.array([idx_to_cls[np.argmax(p)] for p in proba])


def predict_with_thresholds(proba, model_classes, thresholds):
    """
    Remplace l'argmax par une règle de priorité basée sur des seuils abaissés.

    Logique de décision (ordre de priorité décroissant) :
      1. Si P(classe 2 = vital)  >= seuil[2]  → on prédit "vital"    (priorité maximale)
      2. Sinon si P(classe 1 = urgent) >= seuil[1] → on prédit "urgent"
      3. Sinon → on prédit "pas urgent"

    Pourquoi ce mécanisme ?
      Avec l'argmax classique sur 3 classes, un modèle prédit "vital" seulement si P(2) > 33 %.
      En abaissant le seuil à 25 %, on sacrifie un peu de précision pour gagner beaucoup
      de recall sur la classe vitale — ce qui est le bon compromis dans un contexte médical.

    Exemple :
      P = [0.55, 0.30, 0.15] avec seuil[2]=0.10 → prédit 2 (vital)
      P = [0.55, 0.30, 0.05] avec seuil[2]=0.10 → prédit 1 (urgent)
    """
    idx = {cls: i for i, cls in enumerate(model_classes)}  # ex: {0: 0, 1: 1, 2: 2}
    predictions = []
    for p in proba:
        if p[idx[2]] >= thresholds[2]:    # seuil vital atteint ?
            predictions.append(2)
        elif p[idx[1]] >= thresholds[1]:  # seuil urgent atteint ?
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)


def tracer_courbe_seuils(model, model_name, X_test, y_test, thresholds_choisis, save_dir):
    """
    Génère un graphique qui visualise l'impact du seuil de décision sur les métriques.

    On fait varier le seuil de la classe vitale (2) de 0.10 à 0.50 et on trace :
      - le recall de la classe vitale   → baisse quand on monte le seuil (on rate plus d'urgences)
      - la précision de "pas urgent"    → monte quand on monte le seuil (moins de faux positifs)

    Le seuil retenu (ligne verticale grise) est celui qui offre le meilleur compromis
    entre les deux. Cela sert à justifier le choix de 0.25 plutôt qu'une valeur arbitraire.
    """
    proba         = model.predict_proba(X_test)
    model_classes = list(model.classes_)
    seuils        = np.arange(0.10, 0.55, 0.05)  # de 0.10 à 0.50 par pas de 0.05

    recalls_vital         = []
    precisions_pas_urgent = []

    for s in seuils:
        # On garde le seuil de classe 1 fixe, on fait varier uniquement le seuil classe 2
        seuils_test = {2: s, 1: thresholds_choisis[1]}
        y_pred_s    = predict_with_thresholds(proba, model_classes, seuils_test)

        # recall sur classe [2] uniquement
        recalls_vital.append(
            recall_score(y_test, y_pred_s, labels=[2], average=None, zero_division=0)[0]
        )
        # précision sur classe [0] uniquement
        precisions_pas_urgent.append(
            precision_score(y_test, y_pred_s, labels=[0], average=None, zero_division=0)[0]
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seuils, recalls_vital,         label="Recall — Très urgent (cl. 2)",   color="#d62728", marker="o")
    ax.plot(seuils, precisions_pas_urgent, label="Précision — Pas urgent (cl. 0)", color="#1f77b4", marker="s")
    # Ligne verticale : seuil retenu dans la config
    ax.axvline(x=thresholds_choisis[2], color="gray", linestyle="--", linewidth=1.5,
               label=f"Seuil choisi = {thresholds_choisis[2]}")
    ax.set_xlabel("Seuil de décision — classe Très urgent")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Impact du seuil — {model_name}")
    ax.legend()
    plt.tight_layout()
    slug = model_name.replace(" ", "_")
    plt.savefig(os.path.join(save_dir, f"{slug}_threshold_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  → {slug}_threshold_curve.png sauvegardé")


def build_neural_network(input_dim):
    """
    Construit un réseau de neurones multicouches (MLP) pour la classification 3 classes.

    Architecture expliquée couche par couche :
      Dense(256, relu)  → 256 neurones, activation ReLU = max(0, x)
                          ReLU évite le problème du "gradient vanishing" (gradients qui disparaissent)
      Dropout(0.3)      → éteint 30 % des connexions aléatoirement à chaque batch d'entraînement
                          Force le réseau à ne pas sur-dépendre d'un sous-ensemble de neurones → régularisation
      Dense(128, relu)  → couche plus petite : le réseau "compresse" l'information
      Dropout(0.3)
      Dense(3, softmax) → 3 sorties (une par classe), softmax garantit que la somme = 1
                          On obtient directement des probabilités (ex: [0.1, 0.3, 0.6])

    Compilation :
      adam               → optimiseur adaptatif, bonne valeur par défaut
      sparse_categorical_crossentropy → loss pour classification multi-classe avec labels entiers (0, 1, 2)
                                        (pas besoin de one-hot encoding)

    Args:
        input_dim : nombre total de features (colonnes tabulaires + tokens TF-IDF)
    """
    model = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class NNWrapper:
    """
    Adaptateur (pattern "Adapter") pour donner au réseau Keras la même interface que sklearn.

    Le problème : tracer_courbe_seuils appelle model.predict_proba() et model.classes_,
    qui sont des attributs sklearn. Keras n'a pas ces attributs.

    La solution : on enveloppe le modèle Keras dans un objet qui expose les mêmes méthodes,
    sans modifier tracer_courbe_seuils ni le modèle Keras lui-même.

    predict_proba() gère aussi la conversion sparse → dense car Keras ne sait pas lire
    les matrices creuses (scipy.sparse).
    """
    def __init__(self, model):
        self.model    = model
        self.classes_ = np.array([0, 1, 2])  # classes dans l'ordre attendu

    def predict_proba(self, X):
        # scipy.sparse → dense car Keras n'accepte que les tableaux numpy classiques
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        return self.model.predict(X_dense, verbose=0)


# =============================================================================
# CHARGEMENT & PRÉTRAITEMENT (exécuté UNE SEULE FOIS pour les deux configurations)
#
# Pourquoi une seule fois ?
#   Le TF-IDF et le scaler (StandardScaler) sont FIT uniquement sur le train.
#   Si on les recalculait pour chaque config, on perdrait du temps et on risquerait
#   une incohérence. Les deux configs voient exactement les mêmes données prétraitées.
# =============================================================================

df = pd.read_csv(_DATA_PATH)        # chargement du CSV brut
df = apply_business_rules(df)       # suppression des lignes aberrantes (règles métier)

# On filtre les colonnes disponibles (au cas où certaines seraient absentes du CSV)
num_cols = [c for c in _NUM_COL if c in df.columns]
cat_cols = [c for c in _CAT_COL if c in df.columns and c != TEXT_COL]

# ── Split train / test AVANT tout fit ────────────────────────────────────────
# RÈGLE FONDAMENTALE : le TF-IDF et le scaler ne doivent jamais "voir" le test.
# Si on fit d'abord puis on split, le modèle connaît déjà les données de test → biais.
# stratify=df[TARGET] → les proportions des 3 classes sont identiques dans train et test
df_train_raw, df_test_raw = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
)
df_train_raw = df_train_raw.reset_index(drop=True)  # réindexation pour éviter des bugs d'index
df_test_raw  = df_test_raw.reset_index(drop=True)

# ── Prétraitement tabulaire (fit sur train uniquement) ────────────────────────
# preprocessingTechnique retourne (X_transformé, y, pipeline_fit, _)
# Le pipeline_fit contient l'imputer + scaler + encodeur déjà ajustés sur le train.
X_train_tab, y_train, preprocessor, _ = preprocessingTechnique(
    df_train_raw.drop(columns=[TEXT_COL], errors="ignore"),  # on exclut le texte du tabulaire
    target_col=TARGET, num_cols=num_cols, cat_cols=cat_cols, to_drop=_TRASH,
)

# Pour le test : on applique le pipeline FIT sur train (transform only, pas fit_transform)
X_test_raw = df_test_raw.drop(columns=_TRASH + [TARGET, TEXT_COL], errors="ignore")
X_test_tab = preprocessor.transform(X_test_raw)
y_test     = df_test_raw[TARGET].reset_index(drop=True)

# ── TF-IDF sur la description des symptômes ──────────────────────────────────
# TF-IDF = Term Frequency × Inverse Document Frequency
#   → donne plus de poids aux mots rares et significatifs (ex: "douleur thoracique")
#   → et moins de poids aux mots communs inutiles (ex: "le", "une")
#
# Paramètres choisis :
#   max_features=500   → on garde les 500 tokens les plus informatifs (sinon trop large)
#   ngram_range=(1, 2) → on considère les mots seuls ET les paires de mots adjacents
#                        ex: "douleur" + "douleur thoracique" → plus de contexte
#   sublinear_tf=True  → applique log(tf) au lieu de tf brut → réduit l'effet des répétitions
tfidf        = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
X_train_text = tfidf.fit_transform(df_train_raw[TEXT_COL].fillna(""))  # fit + transform sur train
X_test_text  = tfidf.transform(df_test_raw[TEXT_COL].fillna(""))       # transform only sur test

# ── Fusion tabulaire + texte ─────────────────────────────────────────────────
# On concatène les deux matrices horizontalement (côte à côte).
# Problème : X_train_tab est dense (numpy) et X_train_text est creuse (scipy.sparse).
# Solution : on convertit le tabulaire en sparse avant de fusionner avec hstack.
# Format "csr" (Compressed Sparse Row) = format standard pour les opérations ML.
X_train = sp.hstack([sp.csr_matrix(X_train_tab.values), X_train_text], format="csr")
X_test  = sp.hstack([sp.csr_matrix(X_test_tab),         X_test_text],  format="csr")

# Validation croisée stratifiée — utilisée dans evaluate_model
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# =============================================================================
# FONCTION PRINCIPALE : run_scenario(config)
#
# Reçoit une configuration (avec ou sans pénalisation) et exécute le pipeline complet :
#   1. Définit les modèles avec les bons hyperparamètres pour cette config
#   2. Entraîne chaque modèle sklearn + le réseau de neurones
#   3. Génère les prédictions avec la bonne fonction de décision (argmax ou seuils)
#   4. Sauvegarde métriques, matrices, courbes de seuils
#   5. Génère les benchmarks comparatifs entre modèles
# =============================================================================

def run_scenario(config):
    """
    Exécute le pipeline complet pour UNE configuration donnée.

    config : dict avec les clés :
      "nom"          → nom du sous-dossier d'artifacts (ex: "avec_penalisation")
      "class_weight" → dict de poids par classe, ou None
      "thresholds"   → dict de seuils de décision, ou None
    """
    class_weight = config["class_weight"]
    thresholds   = config["thresholds"]
    nom          = config["nom"]

    # Dossier de sortie dédié : artifacts/avec_penalisation/ ou artifacts/sans_penalisation/
    output_dir = os.path.join(ARTIFACTS_BASE, nom)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"  CONFIGURATION : {nom.upper()}")
    print(f"  class_weight  : {class_weight}")
    print(f"  thresholds    : {thresholds}")
    print(f"{'#'*65}")

    # ── Définition des modèles ────────────────────────────────────────────────
    # On les RECRÉE à chaque appel de run_scenario pour repartir d'un état vierge.
    # (les modèles sklearn conservent leurs poids en mémoire après fit)
    #
    # Chaque entrée : nom → (modèle, use_sample_weight, is_penalized)
    #   use_sample_weight : True pour XGBoost qui n'accepte pas class_weight directement
    #   is_penalized      : booléen pour indiquer dans les graphiques si c'est pénalisé
    models = {
        "KNN": (
            KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
            False, False,
            # KNN n'a pas de notion de class_weight → il sert toujours de baseline
            # Le comparer aux autres montre combien la pénalisation change les choses
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE),
            False, class_weight is not None,
            # max_iter=1000 car avec TF-IDF (beaucoup de features), la convergence est plus lente
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, class_weight=class_weight,
                                   random_state=RANDOM_STATE, n_jobs=-1),
            False, class_weight is not None,
            # n_estimators=200 : 200 arbres → bon compromis performance/temps
        ),
        "XGBoost": (
            XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                          random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
            True, class_weight is not None,
            # XGBoost n'a pas de paramètre class_weight natif → on passe sample_weight au fit()
            # use_sample_weight=True signale qu'il faut générer ces poids manuellement
        ),
        "LightGBM": (
            LGBMClassifier(n_estimators=200, class_weight=class_weight,
                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            False, class_weight is not None,
            # verbose=-1 : supprime les messages d'entraînement LightGBM (très verbeux sinon)
        ),
    }

    all_results  = []   # accumule les dicts de métriques de chaque modèle
    all_profiles = []   # accumule les dicts de profiling (temps, RAM, CPU)

    # ── Boucle d'entraînement sklearn ────────────────────────────────────────
    for model_name, (model, use_sample_weight, is_penalized) in models.items():
        print(f"\n{'='*65}")
        print(f"  Entraînement : {model_name}  [{nom}]")
        print(f"{'='*65}")

        # Levier 1 pour XGBoost : traduit class_weight en un vecteur de poids par ligne
        # Ex: si class_weight={0:1, 1:3, 2:10} et y_train=[0, 2, 1, ...]
        #     → sample_weights=[1, 10, 3, ...]
        if use_sample_weight and class_weight:
            sample_weights = np.array([class_weight[int(y)] for y in y_train])
        else:
            sample_weights = None

        # Mesure du temps d'entraînement
        t0 = time.perf_counter()
        if use_sample_weight and sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights)  # XGBoost
        else:
            model.fit(X_train, y_train)  # tous les autres modèles
        train_time_s = time.perf_counter() - t0

        # Levier 2 : prédiction avec seuils personnalisés (config avec pénalisation)
        # ou argmax classique (config sans pénalisation)
        proba  = model.predict_proba(X_test)
        y_pred = predict_fn(proba, list(model.classes_), thresholds)

        # Cross-validation : recall classe 2 sur N_FOLDS folds (stabilité du modèle)
        # On clone le modèle pour éviter de modifier celui déjà entraîné sur le train complet
        cv_recalls2 = []
        for tr_idx, val_idx in cv.split(X_train, y_train):
            X_cv_tr  = X_train[tr_idx]
            X_cv_val = X_train[val_idx]
            y_cv_tr  = y_train.iloc[tr_idx]
            y_cv_val = y_train.iloc[val_idx]
            m_clone  = clone(model)
            if use_sample_weight and class_weight:
                fold_sw = np.array([class_weight[int(y)] for y in y_cv_tr])
                m_clone.fit(X_cv_tr, y_cv_tr, sample_weight=fold_sw)
            else:
                m_clone.fit(X_cv_tr, y_cv_tr)
            fold_proba = m_clone.predict_proba(X_cv_val)
            fold_pred  = predict_fn(fold_proba, list(m_clone.classes_), thresholds)
            cv_recalls2.append(
                recall_score(y_cv_val, fold_pred, labels=[2], average=None, zero_division=0)[0]
            )
        cv_recall2_mean = float(np.mean(cv_recalls2))
        cv_recall2_std  = float(np.std(cv_recalls2))
        print(f"  CV Recall classe 2 ({N_FOLDS} folds) : {cv_recall2_mean:.3f} ± {cv_recall2_std:.3f}")

        # Sous-dossier pour les artifacts de ce modèle dans cette configuration
        model_dir = os.path.join(output_dir, model_name.replace(" ", "_"))
        os.makedirs(model_dir, exist_ok=True)

        # Courbe d'impact du seuil → uniquement dans la config avec pénalisation
        # (inutile de la tracer si on n'utilise pas de seuils)
        if thresholds:
            tracer_courbe_seuils(model, model_name, X_test, y_test, thresholds, model_dir)

        # Calcul des métriques et sauvegarde des PNG (matrice de confusion, tableau)
        result = evaluate_model(
            model_name=model_name,
            history=None,              # pas de courbe de loss pour les modèles sklearn
            y_true=y_test,
            y_pred=y_pred,
            labels=CLASSES,
            label_names=LABEL_NAMES,
            penalized=is_penalized,
            class_weights=class_weight if is_penalized else None,
            output_dir=output_dir,
        )
        all_results.append(result)

        # Mesure des performances techniques (temps d'inférence, RAM au pic, CPU)
        # lambda capture le modèle courant avec m=model pour éviter la fermeture tardive
        profile = profile_model(
            model_name=model_name,
            predict_fn=lambda X, m=model: predict_fn(
                m.predict_proba(X), list(m.classes_), thresholds
            ),
            X_test=X_test,
        )
        profile["train_time_s"] = round(train_time_s, 4)  # ajout du temps d'entraînement
        all_profiles.append(profile)

    # ── Entraînement du réseau de neurones ────────────────────────────────────
    # Traité séparément car l'API Keras est différente de sklearn :
    #   - le fit() prend des arguments différents (epochs, batch_size, validation_split)
    #   - il ne gère pas les matrices creuses (scipy.sparse) → conversion dense obligatoire
    #   - il renvoie un objet History (courbe de loss) qu'on peut tracer
    print(f"\n{'='*65}")
    print(f"  Entraînement : NeuralNetwork  [{nom}]")
    print(f"{'='*65}")

    # Conversion sparse → dense (numpy array) — obligatoire pour Keras
    X_train_dense = X_train.toarray()
    X_test_dense  = X_test.toarray()

    nn_model = build_neural_network(input_dim=X_train_dense.shape[1])

    # EarlyStopping : arrête l'entraînement si la val_loss ne diminue plus pendant 10 epochs
    # restore_best_weights=True → recharge les poids du meilleur epoch (pas forcément le dernier)
    # Évite le sur-apprentissage sans avoir à deviner le bon nombre d'epochs à l'avance
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

    t0 = time.perf_counter()
    nn_history = nn_model.fit(
        X_train_dense, y_train,
        epochs=100,                  # maximum 100 epochs (EarlyStopping arrêtera avant si besoin)
        batch_size=64,               # taille des mini-batchs (compromis vitesse/stabilité)
        validation_split=0.1,        # 10 % du train réservé pour la validation pendant fit
        class_weight=class_weight,   # None si sans_penalisation → comportement standard Keras
        callbacks=[early_stop],
        verbose=1,
    )
    nn_train_time_s = time.perf_counter() - t0

    # Prédictions du réseau de neurones
    nn_proba  = nn_model.predict(X_test_dense, verbose=0)
    nn_y_pred = predict_fn(nn_proba, [0, 1, 2], thresholds)  # classes connues d'avance pour Keras

    nn_model_dir = os.path.join(output_dir, "NeuralNetwork")
    os.makedirs(nn_model_dir, exist_ok=True)

    # NNWrapper pour compatibilité avec tracer_courbe_seuils (interface sklearn)
    nn_wrapper = NNWrapper(nn_model)
    if thresholds:
        tracer_courbe_seuils(nn_wrapper, "NeuralNetwork", X_test, y_test, thresholds, nn_model_dir)

    # history=nn_history déclenche la génération de la courbe de loss dans evaluate_model
    result_nn = evaluate_model(
        model_name="NeuralNetwork",
        history=nn_history,   # ← pour les modèles sklearn c'est None, ici on passe l'historique
        y_true=y_test,
        y_pred=nn_y_pred,
        labels=CLASSES,
        label_names=LABEL_NAMES,
        penalized=class_weight is not None,
        class_weights=class_weight,
        output_dir=output_dir,
    )
    all_results.append(result_nn)

    # Profiling du réseau de neurones (même interface que les modèles sklearn)
    profile_nn = profile_model(
        model_name="NeuralNetwork",
        predict_fn=lambda X: predict_fn(nn_wrapper.predict_proba(X), [0, 1, 2], thresholds),
        X_test=X_test,
    )
    profile_nn["train_time_s"] = round(nn_train_time_s, 4)
    all_profiles.append(profile_nn)

    # ── Génération des benchmarks comparatifs ─────────────────────────────────
    # Ces 4 fonctions génèrent des visuels résumant TOUS les modèles de cette config.
    # Pratique pour comparer d'un coup d'œil KNN vs Random Forest vs réseau de neurones, etc.
    print(f"\nGénération des benchmarks [{nom}]...")
    benchmark_models(all_results,        label_names=LABEL_NAMES, output_dir=output_dir)  # tableau métriques
    benchmark_metrics_chart(all_results, label_names=LABEL_NAMES, output_dir=output_dir)  # barres F1/recall
    benchmark_resources(all_profiles,                             output_dir=output_dir)  # temps & RAM
    benchmark_confusion(all_results,     label_names=LABEL_NAMES, output_dir=output_dir)  # matrices côte à côte
    print(f"  Terminé → artifacts dans {output_dir}")


# =============================================================================
# POINT D'ENTRÉE : exécution des deux configurations
#
# On boucle sur CONFIGURATIONS (avec puis sans pénalisation).
# Les données (X_train, X_test, y_train, y_test) sont déjà prêtes au-dessus.
# Seuls les modèles et les règles de décision changent entre les deux configs.
# =============================================================================

for config in CONFIGURATIONS:
    run_scenario(config)

print("\n\nToutes les configurations terminées.")
print(f"  → artifacts/avec_penalisation/")
print(f"  → artifacts/sans_penalisation/")
