# Scénario 1 – Approche multimodale complète
# Toutes les variables : tabulaire + TF-IDF sur description_symptomes

import sys
import os
import warnings
import numpy as np
import scipy.sparse as sp
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from modules.preprocess import (
    preprocessingTechnique, apply_business_rules,
    _DATA_PATH, _TRASH, _NUM_COL, _CAT_COL,
)
from modules.evaluate import evaluate_model, profile_model, benchmark_models
import pandas as pd

# ── Paramètres ─────────────────────────────────────────────────────────────────
TEXT_COL     = "description_symptomes"
TARGET       = "niveau_urgence"
N_FOLDS      = 5
RANDOM_STATE = 42

# Pénalise fortement le sous-triage vital (classe 2)
CLASS_WEIGHT = {0: 1, 1: 2, 2: 5}

# ── Chargement & règles métier ─────────────────────────────────────────────────
df = pd.read_csv(_DATA_PATH)
df = apply_business_rules(df)

# ── Colonnes tabulaires (filtrées aux colonnes réellement présentes) ───────────
num_cols = [c for c in _NUM_COL if c in df.columns]
cat_cols = [c for c in _CAT_COL if c in df.columns and c != TEXT_COL]

# ── Split stratifié sur données BRUTES (avant tout fit) ──────────────────────
# Obligatoire pour éviter la fuite : TF-IDF et scaler ne doivent voir que le train
df_train_raw, df_test_raw = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
)
df_train_raw = df_train_raw.reset_index(drop=True)
df_test_raw  = df_test_raw.reset_index(drop=True)

# ── Preprocessing tabulaire — fit sur train uniquement ───────────────────────
X_train_tab, y_train, preprocessor, _ = preprocessingTechnique(
    df_train_raw.drop(columns=[TEXT_COL], errors="ignore"),
    target_col=TARGET,
    num_cols=num_cols,
    cat_cols=cat_cols,
    to_drop=_TRASH,
)

# Transform du test avec le preprocessor déjà fitté (pas de re-fit)
X_test_raw = df_test_raw.drop(columns=_TRASH + [TARGET, TEXT_COL], errors="ignore")
X_test_tab = preprocessor.transform(X_test_raw)
y_test = df_test_raw[TARGET].reset_index(drop=True)

# ── TF-IDF — fit sur train uniquement ────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
X_train_text = tfidf.fit_transform(df_train_raw[TEXT_COL].fillna(""))
X_test_text  = tfidf.transform(df_test_raw[TEXT_COL].fillna(""))

# ── Fusion tabulaire (dense → sparse) + texte ────────────────────────────────
X_train = sp.hstack([sp.csr_matrix(X_train_tab.values), X_train_text], format="csr")
X_test  = sp.hstack([sp.csr_matrix(X_test_tab),         X_test_text],  format="csr")

# ── Définition des modèles ────────────────────────────────────────────────────
# KNN n'a pas de class_weight → entraîné sans pondération (référence baseline)
MODELS = {
    "KNN": (
        KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        False,
    ),
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE),
        False,
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=200, class_weight=CLASS_WEIGHT,
                               random_state=RANDOM_STATE, n_jobs=-1),
        False,
    ),
    "XGBoost": (
        XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                      random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        True,   # utilise sample_weight dans fit()
    ),
    "LightGBM": (
        LGBMClassifier(n_estimators=200, class_weight=CLASS_WEIGHT,
                       random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        False,
    ),
}

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

mlflow.set_experiment("scenario_1_complet")

cv_results = {}
print(f"\n{'='*65}")
print(f"  CROSS-VALIDATION ({N_FOLDS} folds) — SCÉNARIO 1 MULTIMODAL")
print(f"{'='*65}")

for name, (model, use_sample_weight) in MODELS.items():
    fold_f1w, fold_f1c2, fold_acc = [], [], []

    with mlflow.start_run(run_name=f"{name}_CV"):
        mlflow.log_param("model",        name)
        mlflow.log_param("n_folds",      N_FOLDS)
        mlflow.log_param("class_weight", CLASS_WEIGHT)
        mlflow.log_param("tfidf_features", 500)

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            fit_kwargs = {}
            if use_sample_weight:
                fit_kwargs["sample_weight"] = compute_sample_weight(CLASS_WEIGHT, y_tr)

            model.fit(X_tr, y_tr, **fit_kwargs)
            y_pred = model.predict(X_val)

            fold_f1w.append( f1_score(y_val, y_pred, average="weighted", zero_division=0))
            fold_f1c2.append(f1_score(y_val, y_pred, average=None, labels=[0,1,2], zero_division=0)[2])
            fold_acc.append( accuracy_score(y_val, y_pred))

        cv_results[name] = {
            "f1_weighted": {"mean": np.mean(fold_f1w),  "std": np.std(fold_f1w)},
            "f1_classe_2": {"mean": np.mean(fold_f1c2), "std": np.std(fold_f1c2)},
            "accuracy":    {"mean": np.mean(fold_acc),  "std": np.std(fold_acc)},
        }

        mlflow.log_metric("cv_f1_weighted_mean", np.mean(fold_f1w))
        mlflow.log_metric("cv_f1_weighted_std",  np.std(fold_f1w))
        mlflow.log_metric("cv_f1_classe2_mean",  np.mean(fold_f1c2))
        mlflow.log_metric("cv_accuracy_mean",    np.mean(fold_acc))

    r = cv_results[name]
    print(f"  {name:<22} "
          f"F1w={r['f1_weighted']['mean']:.4f}±{r['f1_weighted']['std']:.4f}  "
          f"F1_c2={r['f1_classe_2']['mean']:.4f}±{r['f1_classe_2']['std']:.4f}  "
          f"Acc={r['accuracy']['mean']:.4f}")

print(f"{'='*65}\n")

# ── Entraînement final + évaluation sur test ──────────────────────────────────
eval_results  = []
perf_profiles = []

for name, (model, use_sample_weight) in MODELS.items():
    fit_kwargs = {}
    if use_sample_weight:
        fit_kwargs["sample_weight"] = compute_sample_weight(CLASS_WEIGHT, y_train)

    with mlflow.start_run(run_name=f"{name}_final"):
        model.fit(X_train, y_train, **fit_kwargs)
        y_pred = model.predict(X_test)

        result = evaluate_model(name, None, y_test, y_pred, labels=[0, 1, 2])
        eval_results.append(result)

        mlflow.log_metrics({
            "test_accuracy":           result["accuracy"],
            "test_f1_weighted":        result["f1_weighted"],
            "test_f1_classe2":         result["par_classe"][2]["f1"],
            "test_precision_weighted": result["precision_weighted"],
            "test_recall_weighted":    result["recall_weighted"],
        })
        mlflow.sklearn.log_model(model, artifact_path="model")

    perf = profile_model(name, model.predict, X_test)
    perf_profiles.append(perf)

# ── Benchmark final ───────────────────────────────────────────────────────────
df_bench = benchmark_models(eval_results, profiles=perf_profiles, sort_by="f1_classe_2")
