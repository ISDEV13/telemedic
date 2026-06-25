import os, time, warnings, tracemalloc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
import matplotlib
matplotlib.use("Agg")  # pas de fenêtre graphique — on sauvegarde directement en PNG
import matplotlib.pyplot as plt

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
# Ce warning sklearn vient des modèles avec n_jobs=-1 (joblib interne) — sans impact sur les résultats
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Colonnes du dataset ---
_NUM_COL = ["age", "freq_cardiaque", "tension_sys", "temp", "sat_oxygene", "antecedents", "duree_symptomes"]
_CAT_COL = ["sexe", "zone_vie", "source"]
_TXT_COL = "description_symptomes"
_TRASH   = ["patient_id"]
TARGET   = "niveau_urgence"
CLASSES  = [0, 1, 2]
RANDOM_STATE = 42
N_FOLDS      = 5

# Hyperparamètres :
# Pénalisation asymétrique : une erreur vitale coûte 15x plus qu'une erreur non-urgente
CLASS_WEIGHT = {0: 1, 1: 12, 2: 25}
# Seuil vital abaissé à 0.15 : mieux vaut sur-classer que rater un cas vital
THRESHOLDS = {2: 0.10, 1: 0.15}

# Valeurs physiquement impossibles — pas des seuils cliniques, juste des limites absolues
_BORNES = {
    "age":             (0, 130),
    "freq_cardiaque":  (0, 400),
    "tension_sys":     (0, 400),
    "temp":            (0, 60),
    "sat_oxygene":     (0, 100),
    "duree_symptomes": (0, None),
    "antecedents":     (0, None),
}


def run_scenario_complet(data_path, penalize=True):
    """
    Scénario 1 — Multimodal complet : tabulaire + TF-IDF texte.
    Entraîne 5 modèles (LogReg, RandomForest, XGBoost, LightGBM, NeuralNetwork)
    et renvoie une liste de dicts avec leurs performances médicales et techniques.

    Paramètres
    ----------
    data_path : str
        Chemin vers le fichier dataset_telemed.csv.
    penalize : bool
        True  → class_weight asymétrique + seuils abaissés (configuration production).
        False → argmax standard, comportement naturel du modèle (baseline).

    Retour
    ------
    list[dict] — un dict par modèle avec les clés :
        model, model_name,
        recall_class2, recall_class1, precision_class0,
        train_time_s, inference_time_ms_per_sample, ram_peak_mb,
        cv_recall2_mean, cv_recall2_std,
        confusion_matrix
    """

    # =========================================================
    # 1. CHARGEMENT ET RÈGLES MÉTIER
    # =========================================================
    df = pd.read_csv(data_path)

    # Doublons complets supprimés — ils gonflent artificiellement les métriques
    df = df.drop_duplicates()

    # On clippe les valeurs physiquement impossibles mais on conserve les extrêmes réels
    # (SpO2 à 78% ou FC à 180 sont du signal médical utile, pas des erreurs)
    for col, (min_val, max_val) in _BORNES.items():
        if col not in df.columns:
            continue
        df[col] = df[col].clip(lower=min_val, upper=max_val)

    # =========================================================
    # 2. SPLIT STRATIFIÉ
    # =========================================================
    # stratify=TARGET garantit que les 3 classes sont représentées proportionnellement
    # dans le train et le test — indispensable avec des classes déséquilibrées
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df[TARGET].reset_index(drop=True)
    y_test  = test_df[TARGET].reset_index(drop=True)

    # =========================================================
    # 3. PREPROCESSING TABULAIRE (fit sur train uniquement)
    # =========================================================
    # On ne filtre que les colonnes réellement présentes dans le fichier chargé
    num_cols = [c for c in _NUM_COL if c in df.columns]
    cat_cols = [c for c in _CAT_COL if c in df.columns]

    # Médiane robuste aux outliers médicaux, MinMax pour homogénéiser les échelles
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler()),
    ])
    # Mode pour les NA, OneHot pour ne pas introduire d'ordre artificiel entre modalités
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    train_tab = train_df.drop(columns=_TRASH + [TARGET, _TXT_COL], errors="ignore")
    test_tab  = test_df.drop(columns=_TRASH + [TARGET, _TXT_COL], errors="ignore")

    # fit uniquement sur le train — évite toute fuite d'information du test
    X_tab_tr = preprocessor.fit_transform(train_tab)
    X_tab_te = preprocessor.transform(test_tab)

    # =========================================================
    # 4. TF-IDF TEXTE (fit sur train uniquement)
    # =========================================================
    # Bigrammes pour capturer des expressions comme "douleur thoracique"
    # sublinear_tf réduit le poids des termes trop fréquents (ex: "le", "un")
    # Hyperparamètres :
    tfidf    = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
    X_txt_tr = tfidf.fit_transform(train_df[_TXT_COL].fillna(""))
    X_txt_te = tfidf.transform(test_df[_TXT_COL].fillna(""))

    # Fusion tabulaire + TF-IDF — format CSR = Compressed Sparse Row, efficace en mémoire
    X_train = sp.hstack([sp.csr_matrix(X_tab_tr), X_txt_tr], format="csr")
    X_test  = sp.hstack([sp.csr_matrix(X_tab_te), X_txt_te], format="csr")

    # =========================================================
    # 5. PARAMÈTRES DE PÉNALISATION
    # =========================================================
    class_weight = CLASS_WEIGHT if penalize else None
    #MODIF TEMPORAIRE 
    thresholds   = THRESHOLDS   if penalize else None
    # XGBoost gère la pénalisation via sample_weight (pas class_weight comme sklearn)
    xgb_sw = np.array([CLASS_WEIGHT[int(y)] for y in y_train]) if penalize else None

    results = []

    # =========================================================
    # MODÈLE 1 — LOGISTIC REGRESSION
    # =========================================================
    print("[LogisticRegression] Entraînement...")

    # Hyperparamètres :
    model_lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )

    t0_lr = time.perf_counter()
    model_lr.fit(X_train, y_train)
    train_time_lr = time.perf_counter() - t0_lr

    # Prédiction avec seuils asymétriques si penalize=True
    proba_lr = model_lr.predict_proba(X_test)
    if thresholds is not None:
        idx_lr = {cls: i for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = []
        for p in proba_lr:
            # On vérifie le niveau vital en premier : mieux vaut sur-classer que sous-classer
            if p[idx_lr[2]] >= thresholds[2]:
                y_pred_lr.append(2)
            elif p[idx_lr[1]] >= thresholds[1]:
                y_pred_lr.append(1)
            else:
                y_pred_lr.append(0)
        y_pred_lr = np.array(y_pred_lr)
    else:
        idx_to_cls_lr = {i: cls for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = np.array([idx_to_cls_lr[np.argmax(p)] for p in proba_lr])

    # Métriques médicales par classe
    rec_lr  = recall_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    prec_lr = precision_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    cm_lr   = confusion_matrix(y_test, y_pred_lr, labels=CLASSES, normalize="true")

    # Profiling inférence : médiane sur 10 passes pour ignorer les pics ponctuels du système
    durations_lr = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lr.predict_proba(X_test)
        durations_lr.append(time.perf_counter() - t0)
    inference_median_lr     = float(np.median(durations_lr))
    inference_per_sample_lr = inference_median_lr / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lr.predict_proba(X_test)
    _, peak_lr = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lr = peak_lr / 1024 ** 2

    # Cross-validation 5 folds — on mesure la stabilité du recall vital et de la précision non-urgente
    cv_lr = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lr = []
    cv_prec0_lr    = []
    for tr_idx, val_idx in cv_lr.split(X_train, y_train):
        X_cv_tr_lr  = X_train[tr_idx]
        X_cv_val_lr = X_train[val_idx]
        y_cv_tr_lr  = y_train.iloc[tr_idx]
        y_cv_val_lr = y_train.iloc[val_idx]
        m_clone_lr  = clone(model_lr)
        m_clone_lr.fit(X_cv_tr_lr, y_cv_tr_lr)
        fold_proba_lr = m_clone_lr.predict_proba(X_cv_val_lr)
        if thresholds is not None:
            idx_c_lr = {cls: i for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = []
            for p in fold_proba_lr:
                if p[idx_c_lr[2]] >= thresholds[2]:
                    fold_pred_lr.append(2)
                elif p[idx_c_lr[1]] >= thresholds[1]:
                    fold_pred_lr.append(1)
                else:
                    fold_pred_lr.append(0)
            fold_pred_lr = np.array(fold_pred_lr)
        else:
            idx_to_c_lr  = {i: cls for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = np.array([idx_to_c_lr[np.argmax(p)] for p in fold_proba_lr])
        cv_recalls2_lr.append(
            recall_score(y_cv_val_lr, fold_pred_lr, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lr.append(
            precision_score(y_cv_val_lr, fold_pred_lr, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lr,
        "model_name":                   "LogisticRegression",
        "recall_class2":                float(rec_lr[2]),
        "recall_class1":                float(rec_lr[1]),
        "precision_class0":             float(prec_lr[0]),
        "train_time_s":                 round(train_time_lr, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lr, 4),
        "ram_peak_mb":                  round(ram_lr, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lr)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lr)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lr)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lr)), 4),
        "confusion_matrix":             cm_lr.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lr[2]:.3f}  recall_cl1={rec_lr[1]:.3f}  "
        f"prec_cl0={prec_lr[0]:.3f}  train={train_time_lr:.2f}s  "
        f"inférence={inference_per_sample_lr:.4f}ms/éch  ram={ram_lr:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lr):.3f}±{np.std(cv_recalls2_lr):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lr):.3f}±{np.std(cv_prec0_lr):.3f}"
    )

    # =========================================================
    # MODÈLE 2 — RANDOM FOREST
    # =========================================================
    print("[RandomForest] Entraînement...")

    # Hyperparamètres :
    model_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0_rf = time.perf_counter()
    model_rf.fit(X_train, y_train)
    train_time_rf = time.perf_counter() - t0_rf

    proba_rf = model_rf.predict_proba(X_test)
    if thresholds is not None:
        idx_rf = {cls: i for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = []
        for p in proba_rf:
            if p[idx_rf[2]] >= thresholds[2]:
                y_pred_rf.append(2)
            elif p[idx_rf[1]] >= thresholds[1]:
                y_pred_rf.append(1)
            else:
                y_pred_rf.append(0)
        y_pred_rf = np.array(y_pred_rf)
    else:
        idx_to_cls_rf = {i: cls for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = np.array([idx_to_cls_rf[np.argmax(p)] for p in proba_rf])

    rec_rf  = recall_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    prec_rf = precision_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    cm_rf   = confusion_matrix(y_test, y_pred_rf, labels=CLASSES, normalize="true")

    durations_rf = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_rf.predict_proba(X_test)
        durations_rf.append(time.perf_counter() - t0)
    inference_median_rf     = float(np.median(durations_rf))
    inference_per_sample_rf = inference_median_rf / X_test.shape[0] * 1000

    tracemalloc.start()
    model_rf.predict_proba(X_test)
    _, peak_rf = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_rf = peak_rf / 1024 ** 2

    cv_rf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_rf = []
    cv_prec0_rf    = []
    for tr_idx, val_idx in cv_rf.split(X_train, y_train):
        X_cv_tr_rf  = X_train[tr_idx]
        X_cv_val_rf = X_train[val_idx]
        y_cv_tr_rf  = y_train.iloc[tr_idx]
        y_cv_val_rf = y_train.iloc[val_idx]
        m_clone_rf  = clone(model_rf)
        m_clone_rf.fit(X_cv_tr_rf, y_cv_tr_rf)
        fold_proba_rf = m_clone_rf.predict_proba(X_cv_val_rf)
        if thresholds is not None:
            idx_c_rf = {cls: i for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = []
            for p in fold_proba_rf:
                if p[idx_c_rf[2]] >= thresholds[2]:
                    fold_pred_rf.append(2)
                elif p[idx_c_rf[1]] >= thresholds[1]:
                    fold_pred_rf.append(1)
                else:
                    fold_pred_rf.append(0)
            fold_pred_rf = np.array(fold_pred_rf)
        else:
            idx_to_c_rf  = {i: cls for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = np.array([idx_to_c_rf[np.argmax(p)] for p in fold_proba_rf])
        cv_recalls2_rf.append(
            recall_score(y_cv_val_rf, fold_pred_rf, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_rf.append(
            precision_score(y_cv_val_rf, fold_pred_rf, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_rf,
        "model_name":                   "RandomForest",
        "recall_class2":                float(rec_rf[2]),
        "recall_class1":                float(rec_rf[1]),
        "precision_class0":             float(prec_rf[0]),
        "train_time_s":                 round(train_time_rf, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_rf, 4),
        "ram_peak_mb":                  round(ram_rf, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_rf)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_rf)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_rf)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_rf)), 4),
        "confusion_matrix":             cm_rf.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_rf[2]:.3f}  recall_cl1={rec_rf[1]:.3f}  "
        f"prec_cl0={prec_rf[0]:.3f}  train={train_time_rf:.2f}s  "
        f"inférence={inference_per_sample_rf:.4f}ms/éch  ram={ram_rf:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_rf):.3f}±{np.std(cv_recalls2_rf):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_rf):.3f}±{np.std(cv_prec0_rf):.3f}"
    )

    # =========================================================
    # MODÈLE 3 — XGBOOST
    # =========================================================
    print("[XGBoost] Entraînement...")

    # Hyperparamètres :
    model_xgb = XGBClassifier(
        n_estimators=200,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    t0_xgb = time.perf_counter()
    # XGBoost n'accepte pas class_weight directement — on passe sample_weight à fit()
    if xgb_sw is not None:
        model_xgb.fit(X_train, y_train, sample_weight=xgb_sw)
    else:
        model_xgb.fit(X_train, y_train)
    train_time_xgb = time.perf_counter() - t0_xgb

    proba_xgb = model_xgb.predict_proba(X_test)
    if thresholds is not None:
        idx_xgb = {cls: i for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = []
        for p in proba_xgb:
            if p[idx_xgb[2]] >= thresholds[2]:
                y_pred_xgb.append(2)
            elif p[idx_xgb[1]] >= thresholds[1]:
                y_pred_xgb.append(1)
            else:
                y_pred_xgb.append(0)
        y_pred_xgb = np.array(y_pred_xgb)
    else:
        idx_to_cls_xgb = {i: cls for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = np.array([idx_to_cls_xgb[np.argmax(p)] for p in proba_xgb])

    rec_xgb  = recall_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    prec_xgb = precision_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    cm_xgb   = confusion_matrix(y_test, y_pred_xgb, labels=CLASSES, normalize="true")

    durations_xgb = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_xgb.predict_proba(X_test)
        durations_xgb.append(time.perf_counter() - t0)
    inference_median_xgb     = float(np.median(durations_xgb))
    inference_per_sample_xgb = inference_median_xgb / X_test.shape[0] * 1000

    tracemalloc.start()
    model_xgb.predict_proba(X_test)
    _, peak_xgb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_xgb = peak_xgb / 1024 ** 2

    cv_xgb = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_xgb = []
    cv_prec0_xgb    = []
    for tr_idx, val_idx in cv_xgb.split(X_train, y_train):
        X_cv_tr_xgb  = X_train[tr_idx]
        X_cv_val_xgb = X_train[val_idx]
        y_cv_tr_xgb  = y_train.iloc[tr_idx]
        y_cv_val_xgb = y_train.iloc[val_idx]
        m_clone_xgb  = clone(model_xgb)
        if xgb_sw is not None:
            fold_sw_xgb = np.array([CLASS_WEIGHT[int(y)] for y in y_cv_tr_xgb])
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb, sample_weight=fold_sw_xgb)
        else:
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb)
        fold_proba_xgb = m_clone_xgb.predict_proba(X_cv_val_xgb)
        if thresholds is not None:
            idx_c_xgb = {cls: i for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = []
            for p in fold_proba_xgb:
                if p[idx_c_xgb[2]] >= thresholds[2]:
                    fold_pred_xgb.append(2)
                elif p[idx_c_xgb[1]] >= thresholds[1]:
                    fold_pred_xgb.append(1)
                else:
                    fold_pred_xgb.append(0)
            fold_pred_xgb = np.array(fold_pred_xgb)
        else:
            idx_to_c_xgb  = {i: cls for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = np.array([idx_to_c_xgb[np.argmax(p)] for p in fold_proba_xgb])
        cv_recalls2_xgb.append(
            recall_score(y_cv_val_xgb, fold_pred_xgb, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_xgb.append(
            precision_score(y_cv_val_xgb, fold_pred_xgb, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_xgb,
        "model_name":                   "XGBoost",
        "recall_class2":                float(rec_xgb[2]),
        "recall_class1":                float(rec_xgb[1]),
        "precision_class0":             float(prec_xgb[0]),
        "train_time_s":                 round(train_time_xgb, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_xgb, 4),
        "ram_peak_mb":                  round(ram_xgb, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_xgb)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_xgb)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_xgb)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_xgb)), 4),
        "confusion_matrix":             cm_xgb.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_xgb[2]:.3f}  recall_cl1={rec_xgb[1]:.3f}  "
        f"prec_cl0={prec_xgb[0]:.3f}  train={train_time_xgb:.2f}s  "
        f"inférence={inference_per_sample_xgb:.4f}ms/éch  ram={ram_xgb:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_xgb):.3f}±{np.std(cv_recalls2_xgb):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_xgb):.3f}±{np.std(cv_prec0_xgb):.3f}"
    )

    # =========================================================
    # MODÈLE 4 — LIGHTGBM
    # =========================================================
    print("[LightGBM] Entraînement...")

    # Hyperparamètres :
    model_lgbm = LGBMClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    t0_lgbm = time.perf_counter()
    model_lgbm.fit(X_train, y_train)
    train_time_lgbm = time.perf_counter() - t0_lgbm

    proba_lgbm = model_lgbm.predict_proba(X_test)
    if thresholds is not None:
        idx_lgbm = {cls: i for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = []
        for p in proba_lgbm:
            if p[idx_lgbm[2]] >= thresholds[2]:
                y_pred_lgbm.append(2)
            elif p[idx_lgbm[1]] >= thresholds[1]:
                y_pred_lgbm.append(1)
            else:
                y_pred_lgbm.append(0)
        y_pred_lgbm = np.array(y_pred_lgbm)
    else:
        idx_to_cls_lgbm = {i: cls for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = np.array([idx_to_cls_lgbm[np.argmax(p)] for p in proba_lgbm])

    rec_lgbm  = recall_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    prec_lgbm = precision_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    cm_lgbm   = confusion_matrix(y_test, y_pred_lgbm, labels=CLASSES, normalize="true")

    durations_lgbm = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lgbm.predict_proba(X_test)
        durations_lgbm.append(time.perf_counter() - t0)
    inference_median_lgbm     = float(np.median(durations_lgbm))
    inference_per_sample_lgbm = inference_median_lgbm / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lgbm.predict_proba(X_test)
    _, peak_lgbm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lgbm = peak_lgbm / 1024 ** 2

    cv_lgbm = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lgbm = []
    cv_prec0_lgbm    = []
    for tr_idx, val_idx in cv_lgbm.split(X_train, y_train):
        X_cv_tr_lgbm  = X_train[tr_idx]
        X_cv_val_lgbm = X_train[val_idx]
        y_cv_tr_lgbm  = y_train.iloc[tr_idx]
        y_cv_val_lgbm = y_train.iloc[val_idx]
        m_clone_lgbm  = clone(model_lgbm)
        m_clone_lgbm.fit(X_cv_tr_lgbm, y_cv_tr_lgbm)
        fold_proba_lgbm = m_clone_lgbm.predict_proba(X_cv_val_lgbm)
        if thresholds is not None:
            idx_c_lgbm = {cls: i for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = []
            for p in fold_proba_lgbm:
                if p[idx_c_lgbm[2]] >= thresholds[2]:
                    fold_pred_lgbm.append(2)
                elif p[idx_c_lgbm[1]] >= thresholds[1]:
                    fold_pred_lgbm.append(1)
                else:
                    fold_pred_lgbm.append(0)
            fold_pred_lgbm = np.array(fold_pred_lgbm)
        else:
            idx_to_c_lgbm  = {i: cls for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = np.array([idx_to_c_lgbm[np.argmax(p)] for p in fold_proba_lgbm])
        cv_recalls2_lgbm.append(
            recall_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lgbm.append(
            precision_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lgbm,
        "model_name":                   "LightGBM",
        "recall_class2":                float(rec_lgbm[2]),
        "recall_class1":                float(rec_lgbm[1]),
        "precision_class0":             float(prec_lgbm[0]),
        "train_time_s":                 round(train_time_lgbm, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lgbm, 4),
        "ram_peak_mb":                  round(ram_lgbm, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lgbm)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lgbm)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lgbm)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lgbm)), 4),
        "confusion_matrix":             cm_lgbm.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lgbm[2]:.3f}  recall_cl1={rec_lgbm[1]:.3f}  "
        f"prec_cl0={prec_lgbm[0]:.3f}  train={train_time_lgbm:.2f}s  "
        f"inférence={inference_per_sample_lgbm:.4f}ms/éch  ram={ram_lgbm:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lgbm):.3f}±{np.std(cv_recalls2_lgbm):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lgbm):.3f}±{np.std(cv_prec0_lgbm):.3f}"
    )

    # =========================================================
    # MODÈLE 5 — NEURAL NETWORK (Keras/TensorFlow)
    # =========================================================
    print("[NeuralNetwork] Entraînement...")

    # Keras exige des données denses — on convertit la matrice creuse CSR en array numpy
    X_tr_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    X_te_dense = X_test.toarray()  if hasattr(X_test,  "toarray") else np.asarray(X_test)

    # Réseau dense 2 couches cachées, Dropout pour limiter le sur-apprentissage
    # Softmax en sortie → 3 probabilités qui somment à 1
    # Hyperparamètres :
    nn = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(X_tr_dense.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax"),
    ])
    nn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0_nn = time.perf_counter()
    # Hyperparamètres :
    history_nn = nn.fit(
        X_tr_dense, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )],
        verbose=0,
    )
    train_time_nn = time.perf_counter() - t0_nn

    proba_nn = nn.predict(X_te_dense, verbose=0)
    # Pour le NN, les classes sont implicitement [0, 1, 2] dans l'ordre de la sortie softmax
    if thresholds is not None:
        y_pred_nn = []
        for p in proba_nn:
            if p[2] >= thresholds[2]:
                y_pred_nn.append(2)
            elif p[1] >= thresholds[1]:
                y_pred_nn.append(1)
            else:
                y_pred_nn.append(0)
        y_pred_nn = np.array(y_pred_nn)
    else:
        y_pred_nn = np.array([int(np.argmax(p)) for p in proba_nn])

    rec_nn  = recall_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    prec_nn = precision_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    cm_nn   = confusion_matrix(y_test, y_pred_nn, labels=CLASSES, normalize="true")

    durations_nn = []
    for _ in range(10):
        t0 = time.perf_counter()
        nn.predict(X_te_dense, verbose=0)
        durations_nn.append(time.perf_counter() - t0)
    inference_median_nn     = float(np.median(durations_nn))
    inference_per_sample_nn = inference_median_nn / X_test.shape[0] * 1000

    tracemalloc.start()
    nn.predict(X_te_dense, verbose=0)
    _, peak_nn = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_nn = peak_nn / 1024 ** 2

    # La CV n'est pas calculée pour le NN : chaque fold prend autant de temps qu'un entraînement complet
    results.append({
        "model":                        nn,
        "model_name":                   "NeuralNetwork",
        "recall_class2":                float(rec_nn[2]),
        "recall_class1":                float(rec_nn[1]),
        "precision_class0":             float(prec_nn[0]),
        "train_time_s":                 round(train_time_nn, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_nn, 4),
        "ram_peak_mb":                  round(ram_nn, 2),
        "cv_recall2_mean":              None,
        "cv_recall2_std":               None,
        "cv_precision0_mean":           None,
        "cv_precision0_std":            None,
        "confusion_matrix":             cm_nn.tolist(),
        "history":                      history_nn.history,
    })
    print(
        f"  recall_cl2={rec_nn[2]:.3f}  recall_cl1={rec_nn[1]:.3f}  "
        f"prec_cl0={prec_nn[0]:.3f}  train={train_time_nn:.2f}s  "
        f"inférence={inference_per_sample_nn:.4f}ms/éch  ram={ram_nn:.2f}MB  "
        f"CV_recall2=N/A (trop coûteux)"
    )

    return results


def run_scenario_ethique(data_path, penalize=True):
    """
    Scénario 2 — Éthique : tabulaire (sans sexe/zone_vie, age discrétisé) + TF-IDF texte.
    Entraîne 5 modèles et renvoie leurs performances médicales et techniques.

    Différences vs run_scenario_complet :
    - sexe et zone_vie supprimés (conformité RGPD + biais potentiel)
    - age remplacé par une tranche médicale catégorielle (réduction du risque de ré-identification)
    - num_cols réduit en conséquence, cat_cols inclut age discrétisé

    Paramètres
    ----------
    data_path : str
        Chemin vers le fichier dataset_telemed.csv.
    penalize : bool
        True  → class_weight asymétrique + seuils abaissés (configuration production).
        False → argmax standard, comportement naturel du modèle (baseline).

    Retour
    ------
    list[dict] — un dict par modèle avec les clés :
        model, model_name,
        recall_class2, recall_class1, precision_class0,
        train_time_s, inference_time_ms_per_sample, ram_peak_mb,
        cv_recall2_mean, cv_recall2_std, cv_precision0_mean, cv_precision0_std,
        confusion_matrix
    """

    # =========================================================
    # 1. CHARGEMENT ET RÈGLES MÉTIER
    # =========================================================
    df = pd.read_csv(data_path)

    # Doublons complets supprimés — ils gonflent artificiellement les métriques
    df = df.drop_duplicates()

    # On clippe les valeurs physiquement impossibles mais on conserve les extrêmes réels
    for col, (min_val, max_val) in _BORNES.items():
        if col not in df.columns:
            continue
        df[col] = df[col].clip(lower=min_val, upper=max_val)

    # =========================================================
    # 2. PREPROCESSING ÉTHIQUE (avant le split)
    # =========================================================
    # On discrétise l'âge en tranches médicalement significatives pour réduire le risque
    # de ré-identification (un âge précis + constantes vitales = quasi-identifiant)
    _AGE_BINS   = [0, 17, 40, 64, float("inf")]
    _AGE_LABELS = ["enfant", "adulte_jeune", "adulte", "senior"]
    if "age" in df.columns:
        df["age"] = pd.cut(df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=True)

    # On supprime sexe et zone_vie : risque de biais discriminatoire + inutile au sens RGPD
    df = df.drop(columns=["sexe", "zone_vie"], errors="ignore")

    # =========================================================
    # 3. SPLIT STRATIFIÉ
    # =========================================================
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df[TARGET].reset_index(drop=True)
    y_test  = test_df[TARGET].reset_index(drop=True)

    # =========================================================
    # 4. PREPROCESSING TABULAIRE (fit sur train uniquement)
    # =========================================================
    # age est maintenant catégoriel — on l'enlève des num_cols et on l'ajoute aux cat_cols
    _NUM_ETH = ["freq_cardiaque", "tension_sys", "temp", "sat_oxygene", "antecedents", "duree_symptomes"]
    _CAT_ETH = ["source", "age"]  # age discrétisé + source canal de contact

    num_cols = [c for c in _NUM_ETH if c in df.columns]
    cat_cols = [c for c in _CAT_ETH if c in df.columns]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ])

    train_tab = train_df.drop(columns=_TRASH + [TARGET, _TXT_COL], errors="ignore")
    test_tab  = test_df.drop(columns=_TRASH + [TARGET, _TXT_COL], errors="ignore")

    X_tab_tr = preprocessor.fit_transform(train_tab)
    X_tab_te = preprocessor.transform(test_tab)

    # =========================================================
    # 5. TF-IDF TEXTE (fit sur train uniquement)
    # =========================================================
    # Hyperparamètres :
    tfidf    = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
    X_txt_tr = tfidf.fit_transform(train_df[_TXT_COL].fillna(""))
    X_txt_te = tfidf.transform(test_df[_TXT_COL].fillna(""))

    X_train = sp.hstack([sp.csr_matrix(X_tab_tr), X_txt_tr], format="csr")
    X_test  = sp.hstack([sp.csr_matrix(X_tab_te), X_txt_te], format="csr")

    # =========================================================
    # 6. PARAMÈTRES DE PÉNALISATION
    # =========================================================
    class_weight = CLASS_WEIGHT if penalize else None
    
    #MODIF TEMPORAIRE 
    thresholds   = THRESHOLDS   if penalize else None
    xgb_sw = np.array([CLASS_WEIGHT[int(y)] for y in y_train]) if penalize else None

    results = []

    # =========================================================
    # MODÈLE 1 — LOGISTIC REGRESSION
    # =========================================================
    print("[LogisticRegression] Entraînement...")

    # Hyperparamètres :
    model_lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )

    t0_lr = time.perf_counter()
    model_lr.fit(X_train, y_train)
    train_time_lr = time.perf_counter() - t0_lr

    proba_lr = model_lr.predict_proba(X_test)
    if thresholds is not None:
        idx_lr = {cls: i for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = []
        for p in proba_lr:
            if p[idx_lr[2]] >= thresholds[2]:
                y_pred_lr.append(2)
            elif p[idx_lr[1]] >= thresholds[1]:
                y_pred_lr.append(1)
            else:
                y_pred_lr.append(0)
        y_pred_lr = np.array(y_pred_lr)
    else:
        idx_to_cls_lr = {i: cls for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = np.array([idx_to_cls_lr[np.argmax(p)] for p in proba_lr])

    rec_lr  = recall_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    prec_lr = precision_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    cm_lr   = confusion_matrix(y_test, y_pred_lr, labels=CLASSES, normalize="true")

    durations_lr = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lr.predict_proba(X_test)
        durations_lr.append(time.perf_counter() - t0)
    inference_median_lr     = float(np.median(durations_lr))
    inference_per_sample_lr = inference_median_lr / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lr.predict_proba(X_test)
    _, peak_lr = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lr = peak_lr / 1024 ** 2

    cv_lr = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lr = []
    cv_prec0_lr    = []
    for tr_idx, val_idx in cv_lr.split(X_train, y_train):
        X_cv_tr_lr  = X_train[tr_idx]
        X_cv_val_lr = X_train[val_idx]
        y_cv_tr_lr  = y_train.iloc[tr_idx]
        y_cv_val_lr = y_train.iloc[val_idx]
        m_clone_lr  = clone(model_lr)
        m_clone_lr.fit(X_cv_tr_lr, y_cv_tr_lr)
        fold_proba_lr = m_clone_lr.predict_proba(X_cv_val_lr)
        if thresholds is not None:
            idx_c_lr = {cls: i for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = []
            for p in fold_proba_lr:
                if p[idx_c_lr[2]] >= thresholds[2]:
                    fold_pred_lr.append(2)
                elif p[idx_c_lr[1]] >= thresholds[1]:
                    fold_pred_lr.append(1)
                else:
                    fold_pred_lr.append(0)
            fold_pred_lr = np.array(fold_pred_lr)
        else:
            idx_to_c_lr  = {i: cls for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = np.array([idx_to_c_lr[np.argmax(p)] for p in fold_proba_lr])
        cv_recalls2_lr.append(
            recall_score(y_cv_val_lr, fold_pred_lr, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lr.append(
            precision_score(y_cv_val_lr, fold_pred_lr, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lr,
        "model_name":                   "LogisticRegression",
        "recall_class2":                float(rec_lr[2]),
        "recall_class1":                float(rec_lr[1]),
        "precision_class0":             float(prec_lr[0]),
        "train_time_s":                 round(train_time_lr, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lr, 4),
        "ram_peak_mb":                  round(ram_lr, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lr)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lr)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lr)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lr)), 4),
        "confusion_matrix":             cm_lr.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lr[2]:.3f}  recall_cl1={rec_lr[1]:.3f}  "
        f"prec_cl0={prec_lr[0]:.3f}  train={train_time_lr:.2f}s  "
        f"inférence={inference_per_sample_lr:.4f}ms/éch  ram={ram_lr:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lr):.3f}±{np.std(cv_recalls2_lr):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lr):.3f}±{np.std(cv_prec0_lr):.3f}"
    )

    # =========================================================
    # MODÈLE 2 — RANDOM FOREST
    # =========================================================
    print("[RandomForest] Entraînement...")

    # Hyperparamètres :
    model_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0_rf = time.perf_counter()
    model_rf.fit(X_train, y_train)
    train_time_rf = time.perf_counter() - t0_rf

    proba_rf = model_rf.predict_proba(X_test)
    if thresholds is not None:
        idx_rf = {cls: i for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = []
        for p in proba_rf:
            if p[idx_rf[2]] >= thresholds[2]:
                y_pred_rf.append(2)
            elif p[idx_rf[1]] >= thresholds[1]:
                y_pred_rf.append(1)
            else:
                y_pred_rf.append(0)
        y_pred_rf = np.array(y_pred_rf)
    else:
        idx_to_cls_rf = {i: cls for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = np.array([idx_to_cls_rf[np.argmax(p)] for p in proba_rf])

    rec_rf  = recall_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    prec_rf = precision_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    cm_rf   = confusion_matrix(y_test, y_pred_rf, labels=CLASSES, normalize="true")

    durations_rf = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_rf.predict_proba(X_test)
        durations_rf.append(time.perf_counter() - t0)
    inference_median_rf     = float(np.median(durations_rf))
    inference_per_sample_rf = inference_median_rf / X_test.shape[0] * 1000

    tracemalloc.start()
    model_rf.predict_proba(X_test)
    _, peak_rf = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_rf = peak_rf / 1024 ** 2

    cv_rf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_rf = []
    cv_prec0_rf    = []
    for tr_idx, val_idx in cv_rf.split(X_train, y_train):
        X_cv_tr_rf  = X_train[tr_idx]
        X_cv_val_rf = X_train[val_idx]
        y_cv_tr_rf  = y_train.iloc[tr_idx]
        y_cv_val_rf = y_train.iloc[val_idx]
        m_clone_rf  = clone(model_rf)
        m_clone_rf.fit(X_cv_tr_rf, y_cv_tr_rf)
        fold_proba_rf = m_clone_rf.predict_proba(X_cv_val_rf)
        if thresholds is not None:
            idx_c_rf = {cls: i for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = []
            for p in fold_proba_rf:
                if p[idx_c_rf[2]] >= thresholds[2]:
                    fold_pred_rf.append(2)
                elif p[idx_c_rf[1]] >= thresholds[1]:
                    fold_pred_rf.append(1)
                else:
                    fold_pred_rf.append(0)
            fold_pred_rf = np.array(fold_pred_rf)
        else:
            idx_to_c_rf  = {i: cls for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = np.array([idx_to_c_rf[np.argmax(p)] for p in fold_proba_rf])
        cv_recalls2_rf.append(
            recall_score(y_cv_val_rf, fold_pred_rf, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_rf.append(
            precision_score(y_cv_val_rf, fold_pred_rf, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_rf,
        "model_name":                   "RandomForest",
        "recall_class2":                float(rec_rf[2]),
        "recall_class1":                float(rec_rf[1]),
        "precision_class0":             float(prec_rf[0]),
        "train_time_s":                 round(train_time_rf, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_rf, 4),
        "ram_peak_mb":                  round(ram_rf, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_rf)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_rf)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_rf)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_rf)), 4),
        "confusion_matrix":             cm_rf.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_rf[2]:.3f}  recall_cl1={rec_rf[1]:.3f}  "
        f"prec_cl0={prec_rf[0]:.3f}  train={train_time_rf:.2f}s  "
        f"inférence={inference_per_sample_rf:.4f}ms/éch  ram={ram_rf:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_rf):.3f}±{np.std(cv_recalls2_rf):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_rf):.3f}±{np.std(cv_prec0_rf):.3f}"
    )

    # =========================================================
    # MODÈLE 3 — XGBOOST
    # =========================================================
    print("[XGBoost] Entraînement...")

    # Hyperparamètres :
    model_xgb = XGBClassifier(
        n_estimators=200,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    t0_xgb = time.perf_counter()
    if xgb_sw is not None:
        model_xgb.fit(X_train, y_train, sample_weight=xgb_sw)
    else:
        model_xgb.fit(X_train, y_train)
    train_time_xgb = time.perf_counter() - t0_xgb

    proba_xgb = model_xgb.predict_proba(X_test)
    if thresholds is not None:
        idx_xgb = {cls: i for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = []
        for p in proba_xgb:
            if p[idx_xgb[2]] >= thresholds[2]:
                y_pred_xgb.append(2)
            elif p[idx_xgb[1]] >= thresholds[1]:
                y_pred_xgb.append(1)
            else:
                y_pred_xgb.append(0)
        y_pred_xgb = np.array(y_pred_xgb)
    else:
        idx_to_cls_xgb = {i: cls for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = np.array([idx_to_cls_xgb[np.argmax(p)] for p in proba_xgb])

    rec_xgb  = recall_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    prec_xgb = precision_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    cm_xgb   = confusion_matrix(y_test, y_pred_xgb, labels=CLASSES, normalize="true")

    durations_xgb = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_xgb.predict_proba(X_test)
        durations_xgb.append(time.perf_counter() - t0)
    inference_median_xgb     = float(np.median(durations_xgb))
    inference_per_sample_xgb = inference_median_xgb / X_test.shape[0] * 1000

    tracemalloc.start()
    model_xgb.predict_proba(X_test)
    _, peak_xgb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_xgb = peak_xgb / 1024 ** 2

    cv_xgb = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_xgb = []
    cv_prec0_xgb    = []
    for tr_idx, val_idx in cv_xgb.split(X_train, y_train):
        X_cv_tr_xgb  = X_train[tr_idx]
        X_cv_val_xgb = X_train[val_idx]
        y_cv_tr_xgb  = y_train.iloc[tr_idx]
        y_cv_val_xgb = y_train.iloc[val_idx]
        m_clone_xgb  = clone(model_xgb)
        if xgb_sw is not None:
            fold_sw_xgb = np.array([CLASS_WEIGHT[int(y)] for y in y_cv_tr_xgb])
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb, sample_weight=fold_sw_xgb)
        else:
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb)
        fold_proba_xgb = m_clone_xgb.predict_proba(X_cv_val_xgb)
        if thresholds is not None:
            idx_c_xgb = {cls: i for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = []
            for p in fold_proba_xgb:
                if p[idx_c_xgb[2]] >= thresholds[2]:
                    fold_pred_xgb.append(2)
                elif p[idx_c_xgb[1]] >= thresholds[1]:
                    fold_pred_xgb.append(1)
                else:
                    fold_pred_xgb.append(0)
            fold_pred_xgb = np.array(fold_pred_xgb)
        else:
            idx_to_c_xgb  = {i: cls for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = np.array([idx_to_c_xgb[np.argmax(p)] for p in fold_proba_xgb])
        cv_recalls2_xgb.append(
            recall_score(y_cv_val_xgb, fold_pred_xgb, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_xgb.append(
            precision_score(y_cv_val_xgb, fold_pred_xgb, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_xgb,
        "model_name":                   "XGBoost",
        "recall_class2":                float(rec_xgb[2]),
        "recall_class1":                float(rec_xgb[1]),
        "precision_class0":             float(prec_xgb[0]),
        "train_time_s":                 round(train_time_xgb, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_xgb, 4),
        "ram_peak_mb":                  round(ram_xgb, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_xgb)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_xgb)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_xgb)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_xgb)), 4),
        "confusion_matrix":             cm_xgb.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_xgb[2]:.3f}  recall_cl1={rec_xgb[1]:.3f}  "
        f"prec_cl0={prec_xgb[0]:.3f}  train={train_time_xgb:.2f}s  "
        f"inférence={inference_per_sample_xgb:.4f}ms/éch  ram={ram_xgb:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_xgb):.3f}±{np.std(cv_recalls2_xgb):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_xgb):.3f}±{np.std(cv_prec0_xgb):.3f}"
    )

    # =========================================================
    # MODÈLE 4 — LIGHTGBM
    # =========================================================
    print("[LightGBM] Entraînement...")

    # Hyperparamètres :
    model_lgbm = LGBMClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    t0_lgbm = time.perf_counter()
    model_lgbm.fit(X_train, y_train)
    train_time_lgbm = time.perf_counter() - t0_lgbm

    proba_lgbm = model_lgbm.predict_proba(X_test)
    if thresholds is not None:
        idx_lgbm = {cls: i for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = []
        for p in proba_lgbm:
            if p[idx_lgbm[2]] >= thresholds[2]:
                y_pred_lgbm.append(2)
            elif p[idx_lgbm[1]] >= thresholds[1]:
                y_pred_lgbm.append(1)
            else:
                y_pred_lgbm.append(0)
        y_pred_lgbm = np.array(y_pred_lgbm)
    else:
        idx_to_cls_lgbm = {i: cls for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = np.array([idx_to_cls_lgbm[np.argmax(p)] for p in proba_lgbm])

    rec_lgbm  = recall_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    prec_lgbm = precision_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    cm_lgbm   = confusion_matrix(y_test, y_pred_lgbm, labels=CLASSES, normalize="true")

    durations_lgbm = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lgbm.predict_proba(X_test)
        durations_lgbm.append(time.perf_counter() - t0)
    inference_median_lgbm     = float(np.median(durations_lgbm))
    inference_per_sample_lgbm = inference_median_lgbm / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lgbm.predict_proba(X_test)
    _, peak_lgbm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lgbm = peak_lgbm / 1024 ** 2

    cv_lgbm = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lgbm = []
    cv_prec0_lgbm    = []
    for tr_idx, val_idx in cv_lgbm.split(X_train, y_train):
        X_cv_tr_lgbm  = X_train[tr_idx]
        X_cv_val_lgbm = X_train[val_idx]
        y_cv_tr_lgbm  = y_train.iloc[tr_idx]
        y_cv_val_lgbm = y_train.iloc[val_idx]
        m_clone_lgbm  = clone(model_lgbm)
        m_clone_lgbm.fit(X_cv_tr_lgbm, y_cv_tr_lgbm)
        fold_proba_lgbm = m_clone_lgbm.predict_proba(X_cv_val_lgbm)
        if thresholds is not None:
            idx_c_lgbm = {cls: i for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = []
            for p in fold_proba_lgbm:
                if p[idx_c_lgbm[2]] >= thresholds[2]:
                    fold_pred_lgbm.append(2)
                elif p[idx_c_lgbm[1]] >= thresholds[1]:
                    fold_pred_lgbm.append(1)
                else:
                    fold_pred_lgbm.append(0)
            fold_pred_lgbm = np.array(fold_pred_lgbm)
        else:
            idx_to_c_lgbm  = {i: cls for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = np.array([idx_to_c_lgbm[np.argmax(p)] for p in fold_proba_lgbm])
        cv_recalls2_lgbm.append(
            recall_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lgbm.append(
            precision_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lgbm,
        "model_name":                   "LightGBM",
        "recall_class2":                float(rec_lgbm[2]),
        "recall_class1":                float(rec_lgbm[1]),
        "precision_class0":             float(prec_lgbm[0]),
        "train_time_s":                 round(train_time_lgbm, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lgbm, 4),
        "ram_peak_mb":                  round(ram_lgbm, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lgbm)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lgbm)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lgbm)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lgbm)), 4),
        "confusion_matrix":             cm_lgbm.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lgbm[2]:.3f}  recall_cl1={rec_lgbm[1]:.3f}  "
        f"prec_cl0={prec_lgbm[0]:.3f}  train={train_time_lgbm:.2f}s  "
        f"inférence={inference_per_sample_lgbm:.4f}ms/éch  ram={ram_lgbm:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lgbm):.3f}±{np.std(cv_recalls2_lgbm):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lgbm):.3f}±{np.std(cv_prec0_lgbm):.3f}"
    )

    # =========================================================
    # MODÈLE 5 — NEURAL NETWORK (Keras/TensorFlow)
    # =========================================================
    print("[NeuralNetwork] Entraînement...")

    X_tr_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    X_te_dense = X_test.toarray()  if hasattr(X_test,  "toarray") else np.asarray(X_test)

    # Hyperparamètres :
    nn = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(X_tr_dense.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax"),
    ])
    nn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0_nn = time.perf_counter()
    # Hyperparamètres :
    history_nn = nn.fit(
        X_tr_dense, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )],
        verbose=0,
    )
    train_time_nn = time.perf_counter() - t0_nn

    proba_nn = nn.predict(X_te_dense, verbose=0)
    if thresholds is not None:
        y_pred_nn = []
        for p in proba_nn:
            if p[2] >= thresholds[2]:
                y_pred_nn.append(2)
            elif p[1] >= thresholds[1]:
                y_pred_nn.append(1)
            else:
                y_pred_nn.append(0)
        y_pred_nn = np.array(y_pred_nn)
    else:
        y_pred_nn = np.array([int(np.argmax(p)) for p in proba_nn])

    rec_nn  = recall_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    prec_nn = precision_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    cm_nn   = confusion_matrix(y_test, y_pred_nn, labels=CLASSES, normalize="true")

    durations_nn = []
    for _ in range(10):
        t0 = time.perf_counter()
        nn.predict(X_te_dense, verbose=0)
        durations_nn.append(time.perf_counter() - t0)
    inference_median_nn     = float(np.median(durations_nn))
    inference_per_sample_nn = inference_median_nn / X_test.shape[0] * 1000

    tracemalloc.start()
    nn.predict(X_te_dense, verbose=0)
    _, peak_nn = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_nn = peak_nn / 1024 ** 2

    # La CV n'est pas calculée pour le NN : chaque fold prend autant de temps qu'un entraînement complet
    results.append({
        "model":                        nn,
        "model_name":                   "NeuralNetwork",
        "recall_class2":                float(rec_nn[2]),
        "recall_class1":                float(rec_nn[1]),
        "precision_class0":             float(prec_nn[0]),
        "train_time_s":                 round(train_time_nn, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_nn, 4),
        "ram_peak_mb":                  round(ram_nn, 2),
        "cv_recall2_mean":              None,
        "cv_recall2_std":               None,
        "cv_precision0_mean":           None,
        "cv_precision0_std":            None,
        "confusion_matrix":             cm_nn.tolist(),
        "history":                      history_nn.history,
    })
    print(
        f"  recall_cl2={rec_nn[2]:.3f}  recall_cl1={rec_nn[1]:.3f}  "
        f"prec_cl0={prec_nn[0]:.3f}  train={train_time_nn:.2f}s  "
        f"inférence={inference_per_sample_nn:.4f}ms/éch  ram={ram_nn:.2f}MB  "
        f"CV_recall2=N/A (trop coûteux)"
    )

    # On attache le preprocesseur et le tfidf au résultat RF (index 1)
    # pour pouvoir sauvegarder le pipeline complet dans comparer_et_sauvegarder
    results[1]["preprocessor"] = preprocessor
    results[1]["tfidf"]        = tfidf

    return results


def run_scenario_nlp(data_path, penalize=True):
    """
    Scénario 3 — NLP seul : uniquement description_symptomes vectorisée en TF-IDF.
    Aucune constante vitale, aucune variable démographique.
    Mesure si le texte libre seul suffit à prédire l'urgence.

    Paramètres / Retour : identiques à run_scenario_complet.
    """

    # =========================================================
    # 1. CHARGEMENT ET RÈGLES MÉTIER
    # =========================================================
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    for col, (min_val, max_val) in _BORNES.items():
        if col not in df.columns:
            continue
        df[col] = df[col].clip(lower=min_val, upper=max_val)

    # =========================================================
    # 2. SPLIT STRATIFIÉ
    # =========================================================
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df[TARGET].reset_index(drop=True)
    y_test  = test_df[TARGET].reset_index(drop=True)

    # =========================================================
    # 3. TF-IDF UNIQUEMENT (fit sur train uniquement)
    # =========================================================
    # Pas de tabulaire — X_train est directement la matrice TF-IDF
    # On garde les mêmes hyperparamètres que S1 pour la comparabilité
    # Hyperparamètres :
    tfidf   = TfidfVectorizer(max_features=500, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df[_TXT_COL].fillna(""))
    X_test  = tfidf.transform(test_df[_TXT_COL].fillna(""))

    # =========================================================
    # 4. PARAMÈTRES DE PÉNALISATION
    # =========================================================
    class_weight = CLASS_WEIGHT if penalize else None
    thresholds   = THRESHOLDS   if penalize else None
    xgb_sw = np.array([CLASS_WEIGHT[int(y)] for y in y_train]) if penalize else None

    results = []

    # =========================================================
    # MODÈLE 1 — LOGISTIC REGRESSION
    # =========================================================
    print("[LogisticRegression] Entraînement...")

    # Hyperparamètres :
    model_lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )

    t0_lr = time.perf_counter()
    model_lr.fit(X_train, y_train)
    train_time_lr = time.perf_counter() - t0_lr

    proba_lr = model_lr.predict_proba(X_test)
    if thresholds is not None:
        idx_lr = {cls: i for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = []
        for p in proba_lr:
            if p[idx_lr[2]] >= thresholds[2]:
                y_pred_lr.append(2)
            elif p[idx_lr[1]] >= thresholds[1]:
                y_pred_lr.append(1)
            else:
                y_pred_lr.append(0)
        y_pred_lr = np.array(y_pred_lr)
    else:
        idx_to_cls_lr = {i: cls for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = np.array([idx_to_cls_lr[np.argmax(p)] for p in proba_lr])

    rec_lr  = recall_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    prec_lr = precision_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    cm_lr   = confusion_matrix(y_test, y_pred_lr, labels=CLASSES, normalize="true")

    durations_lr = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lr.predict_proba(X_test)
        durations_lr.append(time.perf_counter() - t0)
    inference_median_lr     = float(np.median(durations_lr))
    inference_per_sample_lr = inference_median_lr / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lr.predict_proba(X_test)
    _, peak_lr = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lr = peak_lr / 1024 ** 2

    cv_lr = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lr = []
    cv_prec0_lr    = []
    for tr_idx, val_idx in cv_lr.split(X_train, y_train):
        X_cv_tr_lr  = X_train[tr_idx]
        X_cv_val_lr = X_train[val_idx]
        y_cv_tr_lr  = y_train.iloc[tr_idx]
        y_cv_val_lr = y_train.iloc[val_idx]
        m_clone_lr  = clone(model_lr)
        m_clone_lr.fit(X_cv_tr_lr, y_cv_tr_lr)
        fold_proba_lr = m_clone_lr.predict_proba(X_cv_val_lr)
        if thresholds is not None:
            idx_c_lr = {cls: i for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = []
            for p in fold_proba_lr:
                if p[idx_c_lr[2]] >= thresholds[2]:
                    fold_pred_lr.append(2)
                elif p[idx_c_lr[1]] >= thresholds[1]:
                    fold_pred_lr.append(1)
                else:
                    fold_pred_lr.append(0)
            fold_pred_lr = np.array(fold_pred_lr)
        else:
            idx_to_c_lr  = {i: cls for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = np.array([idx_to_c_lr[np.argmax(p)] for p in fold_proba_lr])
        cv_recalls2_lr.append(
            recall_score(y_cv_val_lr, fold_pred_lr, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lr.append(
            precision_score(y_cv_val_lr, fold_pred_lr, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lr,
        "model_name":                   "LogisticRegression",
        "recall_class2":                float(rec_lr[2]),
        "recall_class1":                float(rec_lr[1]),
        "precision_class0":             float(prec_lr[0]),
        "train_time_s":                 round(train_time_lr, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lr, 4),
        "ram_peak_mb":                  round(ram_lr, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lr)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lr)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lr)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lr)), 4),
        "confusion_matrix":             cm_lr.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lr[2]:.3f}  recall_cl1={rec_lr[1]:.3f}  "
        f"prec_cl0={prec_lr[0]:.3f}  train={train_time_lr:.2f}s  "
        f"inférence={inference_per_sample_lr:.4f}ms/éch  ram={ram_lr:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lr):.3f}±{np.std(cv_recalls2_lr):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lr):.3f}±{np.std(cv_prec0_lr):.3f}"
    )

    # =========================================================
    # MODÈLE 2 — RANDOM FOREST
    # =========================================================
    print("[RandomForest] Entraînement...")

    # Hyperparamètres :
    model_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0_rf = time.perf_counter()
    model_rf.fit(X_train, y_train)
    train_time_rf = time.perf_counter() - t0_rf

    proba_rf = model_rf.predict_proba(X_test)
    if thresholds is not None:
        idx_rf = {cls: i for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = []
        for p in proba_rf:
            if p[idx_rf[2]] >= thresholds[2]:
                y_pred_rf.append(2)
            elif p[idx_rf[1]] >= thresholds[1]:
                y_pred_rf.append(1)
            else:
                y_pred_rf.append(0)
        y_pred_rf = np.array(y_pred_rf)
    else:
        idx_to_cls_rf = {i: cls for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = np.array([idx_to_cls_rf[np.argmax(p)] for p in proba_rf])

    rec_rf  = recall_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    prec_rf = precision_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    cm_rf   = confusion_matrix(y_test, y_pred_rf, labels=CLASSES, normalize="true")

    durations_rf = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_rf.predict_proba(X_test)
        durations_rf.append(time.perf_counter() - t0)
    inference_median_rf     = float(np.median(durations_rf))
    inference_per_sample_rf = inference_median_rf / X_test.shape[0] * 1000

    tracemalloc.start()
    model_rf.predict_proba(X_test)
    _, peak_rf = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_rf = peak_rf / 1024 ** 2

    cv_rf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_rf = []
    cv_prec0_rf    = []
    for tr_idx, val_idx in cv_rf.split(X_train, y_train):
        X_cv_tr_rf  = X_train[tr_idx]
        X_cv_val_rf = X_train[val_idx]
        y_cv_tr_rf  = y_train.iloc[tr_idx]
        y_cv_val_rf = y_train.iloc[val_idx]
        m_clone_rf  = clone(model_rf)
        m_clone_rf.fit(X_cv_tr_rf, y_cv_tr_rf)
        fold_proba_rf = m_clone_rf.predict_proba(X_cv_val_rf)
        if thresholds is not None:
            idx_c_rf = {cls: i for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = []
            for p in fold_proba_rf:
                if p[idx_c_rf[2]] >= thresholds[2]:
                    fold_pred_rf.append(2)
                elif p[idx_c_rf[1]] >= thresholds[1]:
                    fold_pred_rf.append(1)
                else:
                    fold_pred_rf.append(0)
            fold_pred_rf = np.array(fold_pred_rf)
        else:
            idx_to_c_rf  = {i: cls for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = np.array([idx_to_c_rf[np.argmax(p)] for p in fold_proba_rf])
        cv_recalls2_rf.append(
            recall_score(y_cv_val_rf, fold_pred_rf, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_rf.append(
            precision_score(y_cv_val_rf, fold_pred_rf, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_rf,
        "model_name":                   "RandomForest",
        "recall_class2":                float(rec_rf[2]),
        "recall_class1":                float(rec_rf[1]),
        "precision_class0":             float(prec_rf[0]),
        "train_time_s":                 round(train_time_rf, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_rf, 4),
        "ram_peak_mb":                  round(ram_rf, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_rf)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_rf)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_rf)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_rf)), 4),
        "confusion_matrix":             cm_rf.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_rf[2]:.3f}  recall_cl1={rec_rf[1]:.3f}  "
        f"prec_cl0={prec_rf[0]:.3f}  train={train_time_rf:.2f}s  "
        f"inférence={inference_per_sample_rf:.4f}ms/éch  ram={ram_rf:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_rf):.3f}±{np.std(cv_recalls2_rf):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_rf):.3f}±{np.std(cv_prec0_rf):.3f}"
    )

    # =========================================================
    # MODÈLE 3 — XGBOOST
    # =========================================================
    print("[XGBoost] Entraînement...")

    # Hyperparamètres :
    model_xgb = XGBClassifier(
        n_estimators=200,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    t0_xgb = time.perf_counter()
    if xgb_sw is not None:
        model_xgb.fit(X_train, y_train, sample_weight=xgb_sw)
    else:
        model_xgb.fit(X_train, y_train)
    train_time_xgb = time.perf_counter() - t0_xgb

    proba_xgb = model_xgb.predict_proba(X_test)
    if thresholds is not None:
        idx_xgb = {cls: i for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = []
        for p in proba_xgb:
            if p[idx_xgb[2]] >= thresholds[2]:
                y_pred_xgb.append(2)
            elif p[idx_xgb[1]] >= thresholds[1]:
                y_pred_xgb.append(1)
            else:
                y_pred_xgb.append(0)
        y_pred_xgb = np.array(y_pred_xgb)
    else:
        idx_to_cls_xgb = {i: cls for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = np.array([idx_to_cls_xgb[np.argmax(p)] for p in proba_xgb])

    rec_xgb  = recall_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    prec_xgb = precision_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    cm_xgb   = confusion_matrix(y_test, y_pred_xgb, labels=CLASSES, normalize="true")

    durations_xgb = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_xgb.predict_proba(X_test)
        durations_xgb.append(time.perf_counter() - t0)
    inference_median_xgb     = float(np.median(durations_xgb))
    inference_per_sample_xgb = inference_median_xgb / X_test.shape[0] * 1000

    tracemalloc.start()
    model_xgb.predict_proba(X_test)
    _, peak_xgb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_xgb = peak_xgb / 1024 ** 2

    cv_xgb = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_xgb = []
    cv_prec0_xgb    = []
    for tr_idx, val_idx in cv_xgb.split(X_train, y_train):
        X_cv_tr_xgb  = X_train[tr_idx]
        X_cv_val_xgb = X_train[val_idx]
        y_cv_tr_xgb  = y_train.iloc[tr_idx]
        y_cv_val_xgb = y_train.iloc[val_idx]
        m_clone_xgb  = clone(model_xgb)
        if xgb_sw is not None:
            fold_sw_xgb = np.array([CLASS_WEIGHT[int(y)] for y in y_cv_tr_xgb])
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb, sample_weight=fold_sw_xgb)
        else:
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb)
        fold_proba_xgb = m_clone_xgb.predict_proba(X_cv_val_xgb)
        if thresholds is not None:
            idx_c_xgb = {cls: i for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = []
            for p in fold_proba_xgb:
                if p[idx_c_xgb[2]] >= thresholds[2]:
                    fold_pred_xgb.append(2)
                elif p[idx_c_xgb[1]] >= thresholds[1]:
                    fold_pred_xgb.append(1)
                else:
                    fold_pred_xgb.append(0)
            fold_pred_xgb = np.array(fold_pred_xgb)
        else:
            idx_to_c_xgb  = {i: cls for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = np.array([idx_to_c_xgb[np.argmax(p)] for p in fold_proba_xgb])
        cv_recalls2_xgb.append(
            recall_score(y_cv_val_xgb, fold_pred_xgb, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_xgb.append(
            precision_score(y_cv_val_xgb, fold_pred_xgb, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_xgb,
        "model_name":                   "XGBoost",
        "recall_class2":                float(rec_xgb[2]),
        "recall_class1":                float(rec_xgb[1]),
        "precision_class0":             float(prec_xgb[0]),
        "train_time_s":                 round(train_time_xgb, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_xgb, 4),
        "ram_peak_mb":                  round(ram_xgb, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_xgb)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_xgb)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_xgb)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_xgb)), 4),
        "confusion_matrix":             cm_xgb.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_xgb[2]:.3f}  recall_cl1={rec_xgb[1]:.3f}  "
        f"prec_cl0={prec_xgb[0]:.3f}  train={train_time_xgb:.2f}s  "
        f"inférence={inference_per_sample_xgb:.4f}ms/éch  ram={ram_xgb:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_xgb):.3f}±{np.std(cv_recalls2_xgb):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_xgb):.3f}±{np.std(cv_prec0_xgb):.3f}"
    )

    # =========================================================
    # MODÈLE 4 — LIGHTGBM
    # =========================================================
    print("[LightGBM] Entraînement...")

    # Hyperparamètres :
    model_lgbm = LGBMClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    t0_lgbm = time.perf_counter()
    model_lgbm.fit(X_train, y_train)
    train_time_lgbm = time.perf_counter() - t0_lgbm

    proba_lgbm = model_lgbm.predict_proba(X_test)
    if thresholds is not None:
        idx_lgbm = {cls: i for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = []
        for p in proba_lgbm:
            if p[idx_lgbm[2]] >= thresholds[2]:
                y_pred_lgbm.append(2)
            elif p[idx_lgbm[1]] >= thresholds[1]:
                y_pred_lgbm.append(1)
            else:
                y_pred_lgbm.append(0)
        y_pred_lgbm = np.array(y_pred_lgbm)
    else:
        idx_to_cls_lgbm = {i: cls for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = np.array([idx_to_cls_lgbm[np.argmax(p)] for p in proba_lgbm])

    rec_lgbm  = recall_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    prec_lgbm = precision_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    cm_lgbm   = confusion_matrix(y_test, y_pred_lgbm, labels=CLASSES, normalize="true")

    durations_lgbm = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lgbm.predict_proba(X_test)
        durations_lgbm.append(time.perf_counter() - t0)
    inference_median_lgbm     = float(np.median(durations_lgbm))
    inference_per_sample_lgbm = inference_median_lgbm / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lgbm.predict_proba(X_test)
    _, peak_lgbm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lgbm = peak_lgbm / 1024 ** 2

    cv_lgbm = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lgbm = []
    cv_prec0_lgbm    = []
    for tr_idx, val_idx in cv_lgbm.split(X_train, y_train):
        X_cv_tr_lgbm  = X_train[tr_idx]
        X_cv_val_lgbm = X_train[val_idx]
        y_cv_tr_lgbm  = y_train.iloc[tr_idx]
        y_cv_val_lgbm = y_train.iloc[val_idx]
        m_clone_lgbm  = clone(model_lgbm)
        m_clone_lgbm.fit(X_cv_tr_lgbm, y_cv_tr_lgbm)
        fold_proba_lgbm = m_clone_lgbm.predict_proba(X_cv_val_lgbm)
        if thresholds is not None:
            idx_c_lgbm = {cls: i for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = []
            for p in fold_proba_lgbm:
                if p[idx_c_lgbm[2]] >= thresholds[2]:
                    fold_pred_lgbm.append(2)
                elif p[idx_c_lgbm[1]] >= thresholds[1]:
                    fold_pred_lgbm.append(1)
                else:
                    fold_pred_lgbm.append(0)
            fold_pred_lgbm = np.array(fold_pred_lgbm)
        else:
            idx_to_c_lgbm  = {i: cls for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = np.array([idx_to_c_lgbm[np.argmax(p)] for p in fold_proba_lgbm])
        cv_recalls2_lgbm.append(
            recall_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lgbm.append(
            precision_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lgbm,
        "model_name":                   "LightGBM",
        "recall_class2":                float(rec_lgbm[2]),
        "recall_class1":                float(rec_lgbm[1]),
        "precision_class0":             float(prec_lgbm[0]),
        "train_time_s":                 round(train_time_lgbm, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lgbm, 4),
        "ram_peak_mb":                  round(ram_lgbm, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lgbm)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lgbm)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lgbm)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lgbm)), 4),
        "confusion_matrix":             cm_lgbm.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lgbm[2]:.3f}  recall_cl1={rec_lgbm[1]:.3f}  "
        f"prec_cl0={prec_lgbm[0]:.3f}  train={train_time_lgbm:.2f}s  "
        f"inférence={inference_per_sample_lgbm:.4f}ms/éch  ram={ram_lgbm:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lgbm):.3f}±{np.std(cv_recalls2_lgbm):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lgbm):.3f}±{np.std(cv_prec0_lgbm):.3f}"
    )

    # =========================================================
    # MODÈLE 5 — NEURAL NETWORK (Keras/TensorFlow)
    # =========================================================
    print("[NeuralNetwork] Entraînement...")

    # TF-IDF produit déjà une matrice creuse — conversion en dense pour Keras
    X_tr_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    X_te_dense = X_test.toarray()  if hasattr(X_test,  "toarray") else np.asarray(X_test)

    # Hyperparamètres :
    nn = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(X_tr_dense.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax"),
    ])
    nn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0_nn = time.perf_counter()
    # Hyperparamètres :
    history_nn = nn.fit(
        X_tr_dense, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )],
        verbose=0,
    )
    train_time_nn = time.perf_counter() - t0_nn

    proba_nn = nn.predict(X_te_dense, verbose=0)
    if thresholds is not None:
        y_pred_nn = []
        for p in proba_nn:
            if p[2] >= thresholds[2]:
                y_pred_nn.append(2)
            elif p[1] >= thresholds[1]:
                y_pred_nn.append(1)
            else:
                y_pred_nn.append(0)
        y_pred_nn = np.array(y_pred_nn)
    else:
        y_pred_nn = np.array([int(np.argmax(p)) for p in proba_nn])

    rec_nn  = recall_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    prec_nn = precision_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    cm_nn   = confusion_matrix(y_test, y_pred_nn, labels=CLASSES, normalize="true")

    durations_nn = []
    for _ in range(10):
        t0 = time.perf_counter()
        nn.predict(X_te_dense, verbose=0)
        durations_nn.append(time.perf_counter() - t0)
    inference_median_nn     = float(np.median(durations_nn))
    inference_per_sample_nn = inference_median_nn / X_test.shape[0] * 1000

    tracemalloc.start()
    nn.predict(X_te_dense, verbose=0)
    _, peak_nn = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_nn = peak_nn / 1024 ** 2

    results.append({
        "model":                        nn,
        "model_name":                   "NeuralNetwork",
        "recall_class2":                float(rec_nn[2]),
        "recall_class1":                float(rec_nn[1]),
        "precision_class0":             float(prec_nn[0]),
        "train_time_s":                 round(train_time_nn, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_nn, 4),
        "ram_peak_mb":                  round(ram_nn, 2),
        "cv_recall2_mean":              None,
        "cv_recall2_std":               None,
        "cv_precision0_mean":           None,
        "cv_precision0_std":            None,
        "confusion_matrix":             cm_nn.tolist(),
        "history":                      history_nn.history,
    })
    print(
        f"  recall_cl2={rec_nn[2]:.3f}  recall_cl1={rec_nn[1]:.3f}  "
        f"prec_cl0={prec_nn[0]:.3f}  train={train_time_nn:.2f}s  "
        f"inférence={inference_per_sample_nn:.4f}ms/éch  ram={ram_nn:.2f}MB  "
        f"CV_recall2=N/A (trop coûteux)"
    )

    return results


def run_scenario_clinique(data_path, penalize=True):
    """
    Scénario 4 — Clinique pur : uniquement les constantes vitales numériques + âge.
    Ni texte, ni catégorielles, ni canal de contact.
    Simule une fiche clinique automatique sans description textuelle.

    Paramètres / Retour : identiques à run_scenario_complet.
    """

    # =========================================================
    # 1. CHARGEMENT ET RÈGLES MÉTIER
    # =========================================================
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    for col, (min_val, max_val) in _BORNES.items():
        if col not in df.columns:
            continue
        df[col] = df[col].clip(lower=min_val, upper=max_val)

    # =========================================================
    # 2. SPLIT STRATIFIÉ
    # =========================================================
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    y_train = train_df[TARGET].reset_index(drop=True)
    y_test  = test_df[TARGET].reset_index(drop=True)

    # =========================================================
    # 3. PREPROCESSING TABULAIRE CLINIQUE (fit sur train uniquement)
    # =========================================================
    # On garde uniquement les variables physiologiques pures — pas de texte, pas de catégorielles
    # Une tachycardie seule peut être une crise d'angoisse ou un choc septique :
    # ce scénario mesure jusqu'où les vitaux seuls peuvent aller
    _NUM_CLI = ["age", "freq_cardiaque", "tension_sys", "temp", "sat_oxygene", "antecedents", "duree_symptomes"]
    num_cols = [c for c in _NUM_CLI if c in df.columns]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  MinMaxScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
    ])

    # On retire tout ce qui n'est pas numérique clinique
    cols_a_garder = num_cols + [TARGET]
    train_tab = train_df[[c for c in cols_a_garder if c in train_df.columns]]
    test_tab  = test_df[[c for c in cols_a_garder if c in test_df.columns]]

    X_tab_tr = preprocessor.fit_transform(train_tab.drop(columns=[TARGET]))
    X_tab_te = preprocessor.transform(test_tab.drop(columns=[TARGET]))

    # Pas de TF-IDF ni de texte — X_train est directement la matrice tabulaire dense
    X_train = X_tab_tr
    X_test  = X_tab_te

    # =========================================================
    # 4. PARAMÈTRES DE PÉNALISATION
    # =========================================================
    class_weight = CLASS_WEIGHT if penalize else None
    thresholds   = THRESHOLDS   if penalize else None
    xgb_sw = np.array([CLASS_WEIGHT[int(y)] for y in y_train]) if penalize else None

    results = []

    # =========================================================
    # MODÈLE 1 — LOGISTIC REGRESSION
    # =========================================================
    print("[LogisticRegression] Entraînement...")

    # Hyperparamètres :
    model_lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )

    t0_lr = time.perf_counter()
    model_lr.fit(X_train, y_train)
    train_time_lr = time.perf_counter() - t0_lr

    proba_lr = model_lr.predict_proba(X_test)
    if thresholds is not None:
        idx_lr = {cls: i for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = []
        for p in proba_lr:
            if p[idx_lr[2]] >= thresholds[2]:
                y_pred_lr.append(2)
            elif p[idx_lr[1]] >= thresholds[1]:
                y_pred_lr.append(1)
            else:
                y_pred_lr.append(0)
        y_pred_lr = np.array(y_pred_lr)
    else:
        idx_to_cls_lr = {i: cls for i, cls in enumerate(model_lr.classes_)}
        y_pred_lr = np.array([idx_to_cls_lr[np.argmax(p)] for p in proba_lr])

    rec_lr  = recall_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    prec_lr = precision_score(y_test, y_pred_lr, average=None, labels=CLASSES, zero_division=0)
    cm_lr   = confusion_matrix(y_test, y_pred_lr, labels=CLASSES, normalize="true")

    durations_lr = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lr.predict_proba(X_test)
        durations_lr.append(time.perf_counter() - t0)
    inference_median_lr     = float(np.median(durations_lr))
    inference_per_sample_lr = inference_median_lr / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lr.predict_proba(X_test)
    _, peak_lr = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lr = peak_lr / 1024 ** 2

    cv_lr = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lr = []
    cv_prec0_lr    = []
    for tr_idx, val_idx in cv_lr.split(X_train, y_train):
        X_cv_tr_lr  = X_train[tr_idx]
        X_cv_val_lr = X_train[val_idx]
        y_cv_tr_lr  = y_train.iloc[tr_idx]
        y_cv_val_lr = y_train.iloc[val_idx]
        m_clone_lr  = clone(model_lr)
        m_clone_lr.fit(X_cv_tr_lr, y_cv_tr_lr)
        fold_proba_lr = m_clone_lr.predict_proba(X_cv_val_lr)
        if thresholds is not None:
            idx_c_lr = {cls: i for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = []
            for p in fold_proba_lr:
                if p[idx_c_lr[2]] >= thresholds[2]:
                    fold_pred_lr.append(2)
                elif p[idx_c_lr[1]] >= thresholds[1]:
                    fold_pred_lr.append(1)
                else:
                    fold_pred_lr.append(0)
            fold_pred_lr = np.array(fold_pred_lr)
        else:
            idx_to_c_lr  = {i: cls for i, cls in enumerate(m_clone_lr.classes_)}
            fold_pred_lr = np.array([idx_to_c_lr[np.argmax(p)] for p in fold_proba_lr])
        cv_recalls2_lr.append(
            recall_score(y_cv_val_lr, fold_pred_lr, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lr.append(
            precision_score(y_cv_val_lr, fold_pred_lr, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lr,
        "model_name":                   "LogisticRegression",
        "recall_class2":                float(rec_lr[2]),
        "recall_class1":                float(rec_lr[1]),
        "precision_class0":             float(prec_lr[0]),
        "train_time_s":                 round(train_time_lr, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lr, 4),
        "ram_peak_mb":                  round(ram_lr, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lr)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lr)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lr)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lr)), 4),
        "confusion_matrix":             cm_lr.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lr[2]:.3f}  recall_cl1={rec_lr[1]:.3f}  "
        f"prec_cl0={prec_lr[0]:.3f}  train={train_time_lr:.2f}s  "
        f"inférence={inference_per_sample_lr:.4f}ms/éch  ram={ram_lr:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lr):.3f}±{np.std(cv_recalls2_lr):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lr):.3f}±{np.std(cv_prec0_lr):.3f}"
    )

    # =========================================================
    # MODÈLE 2 — RANDOM FOREST
    # =========================================================
    print("[RandomForest] Entraînement...")

    # Hyperparamètres :
    model_rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    t0_rf = time.perf_counter()
    model_rf.fit(X_train, y_train)
    train_time_rf = time.perf_counter() - t0_rf

    proba_rf = model_rf.predict_proba(X_test)
    if thresholds is not None:
        idx_rf = {cls: i for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = []
        for p in proba_rf:
            if p[idx_rf[2]] >= thresholds[2]:
                y_pred_rf.append(2)
            elif p[idx_rf[1]] >= thresholds[1]:
                y_pred_rf.append(1)
            else:
                y_pred_rf.append(0)
        y_pred_rf = np.array(y_pred_rf)
    else:
        idx_to_cls_rf = {i: cls for i, cls in enumerate(model_rf.classes_)}
        y_pred_rf = np.array([idx_to_cls_rf[np.argmax(p)] for p in proba_rf])

    rec_rf  = recall_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    prec_rf = precision_score(y_test, y_pred_rf, average=None, labels=CLASSES, zero_division=0)
    cm_rf   = confusion_matrix(y_test, y_pred_rf, labels=CLASSES, normalize="true")

    durations_rf = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_rf.predict_proba(X_test)
        durations_rf.append(time.perf_counter() - t0)
    inference_median_rf     = float(np.median(durations_rf))
    inference_per_sample_rf = inference_median_rf / X_test.shape[0] * 1000

    tracemalloc.start()
    model_rf.predict_proba(X_test)
    _, peak_rf = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_rf = peak_rf / 1024 ** 2

    cv_rf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_rf = []
    cv_prec0_rf    = []
    for tr_idx, val_idx in cv_rf.split(X_train, y_train):
        X_cv_tr_rf  = X_train[tr_idx]
        X_cv_val_rf = X_train[val_idx]
        y_cv_tr_rf  = y_train.iloc[tr_idx]
        y_cv_val_rf = y_train.iloc[val_idx]
        m_clone_rf  = clone(model_rf)
        m_clone_rf.fit(X_cv_tr_rf, y_cv_tr_rf)
        fold_proba_rf = m_clone_rf.predict_proba(X_cv_val_rf)
        if thresholds is not None:
            idx_c_rf = {cls: i for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = []
            for p in fold_proba_rf:
                if p[idx_c_rf[2]] >= thresholds[2]:
                    fold_pred_rf.append(2)
                elif p[idx_c_rf[1]] >= thresholds[1]:
                    fold_pred_rf.append(1)
                else:
                    fold_pred_rf.append(0)
            fold_pred_rf = np.array(fold_pred_rf)
        else:
            idx_to_c_rf  = {i: cls for i, cls in enumerate(m_clone_rf.classes_)}
            fold_pred_rf = np.array([idx_to_c_rf[np.argmax(p)] for p in fold_proba_rf])
        cv_recalls2_rf.append(
            recall_score(y_cv_val_rf, fold_pred_rf, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_rf.append(
            precision_score(y_cv_val_rf, fold_pred_rf, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_rf,
        "model_name":                   "RandomForest",
        "recall_class2":                float(rec_rf[2]),
        "recall_class1":                float(rec_rf[1]),
        "precision_class0":             float(prec_rf[0]),
        "train_time_s":                 round(train_time_rf, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_rf, 4),
        "ram_peak_mb":                  round(ram_rf, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_rf)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_rf)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_rf)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_rf)), 4),
        "confusion_matrix":             cm_rf.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_rf[2]:.3f}  recall_cl1={rec_rf[1]:.3f}  "
        f"prec_cl0={prec_rf[0]:.3f}  train={train_time_rf:.2f}s  "
        f"inférence={inference_per_sample_rf:.4f}ms/éch  ram={ram_rf:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_rf):.3f}±{np.std(cv_recalls2_rf):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_rf):.3f}±{np.std(cv_prec0_rf):.3f}"
    )

    # =========================================================
    # MODÈLE 3 — XGBOOST
    # =========================================================
    print("[XGBoost] Entraînement...")

    # Hyperparamètres :
    model_xgb = XGBClassifier(
        n_estimators=200,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    t0_xgb = time.perf_counter()
    if xgb_sw is not None:
        model_xgb.fit(X_train, y_train, sample_weight=xgb_sw)
    else:
        model_xgb.fit(X_train, y_train)
    train_time_xgb = time.perf_counter() - t0_xgb

    proba_xgb = model_xgb.predict_proba(X_test)
    if thresholds is not None:
        idx_xgb = {cls: i for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = []
        for p in proba_xgb:
            if p[idx_xgb[2]] >= thresholds[2]:
                y_pred_xgb.append(2)
            elif p[idx_xgb[1]] >= thresholds[1]:
                y_pred_xgb.append(1)
            else:
                y_pred_xgb.append(0)
        y_pred_xgb = np.array(y_pred_xgb)
    else:
        idx_to_cls_xgb = {i: cls for i, cls in enumerate(model_xgb.classes_)}
        y_pred_xgb = np.array([idx_to_cls_xgb[np.argmax(p)] for p in proba_xgb])

    rec_xgb  = recall_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    prec_xgb = precision_score(y_test, y_pred_xgb, average=None, labels=CLASSES, zero_division=0)
    cm_xgb   = confusion_matrix(y_test, y_pred_xgb, labels=CLASSES, normalize="true")

    durations_xgb = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_xgb.predict_proba(X_test)
        durations_xgb.append(time.perf_counter() - t0)
    inference_median_xgb     = float(np.median(durations_xgb))
    inference_per_sample_xgb = inference_median_xgb / X_test.shape[0] * 1000

    tracemalloc.start()
    model_xgb.predict_proba(X_test)
    _, peak_xgb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_xgb = peak_xgb / 1024 ** 2

    cv_xgb = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_xgb = []
    cv_prec0_xgb    = []
    for tr_idx, val_idx in cv_xgb.split(X_train, y_train):
        X_cv_tr_xgb  = X_train[tr_idx]
        X_cv_val_xgb = X_train[val_idx]
        y_cv_tr_xgb  = y_train.iloc[tr_idx]
        y_cv_val_xgb = y_train.iloc[val_idx]
        m_clone_xgb  = clone(model_xgb)
        if xgb_sw is not None:
            fold_sw_xgb = np.array([CLASS_WEIGHT[int(y)] for y in y_cv_tr_xgb])
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb, sample_weight=fold_sw_xgb)
        else:
            m_clone_xgb.fit(X_cv_tr_xgb, y_cv_tr_xgb)
        fold_proba_xgb = m_clone_xgb.predict_proba(X_cv_val_xgb)
        if thresholds is not None:
            idx_c_xgb = {cls: i for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = []
            for p in fold_proba_xgb:
                if p[idx_c_xgb[2]] >= thresholds[2]:
                    fold_pred_xgb.append(2)
                elif p[idx_c_xgb[1]] >= thresholds[1]:
                    fold_pred_xgb.append(1)
                else:
                    fold_pred_xgb.append(0)
            fold_pred_xgb = np.array(fold_pred_xgb)
        else:
            idx_to_c_xgb  = {i: cls for i, cls in enumerate(m_clone_xgb.classes_)}
            fold_pred_xgb = np.array([idx_to_c_xgb[np.argmax(p)] for p in fold_proba_xgb])
        cv_recalls2_xgb.append(
            recall_score(y_cv_val_xgb, fold_pred_xgb, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_xgb.append(
            precision_score(y_cv_val_xgb, fold_pred_xgb, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_xgb,
        "model_name":                   "XGBoost",
        "recall_class2":                float(rec_xgb[2]),
        "recall_class1":                float(rec_xgb[1]),
        "precision_class0":             float(prec_xgb[0]),
        "train_time_s":                 round(train_time_xgb, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_xgb, 4),
        "ram_peak_mb":                  round(ram_xgb, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_xgb)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_xgb)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_xgb)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_xgb)), 4),
        "confusion_matrix":             cm_xgb.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_xgb[2]:.3f}  recall_cl1={rec_xgb[1]:.3f}  "
        f"prec_cl0={prec_xgb[0]:.3f}  train={train_time_xgb:.2f}s  "
        f"inférence={inference_per_sample_xgb:.4f}ms/éch  ram={ram_xgb:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_xgb):.3f}±{np.std(cv_recalls2_xgb):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_xgb):.3f}±{np.std(cv_prec0_xgb):.3f}"
    )

    # =========================================================
    # MODÈLE 4 — LIGHTGBM
    # =========================================================
    print("[LightGBM] Entraînement...")

    # Hyperparamètres :
    model_lgbm = LGBMClassifier(
        n_estimators=200,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    t0_lgbm = time.perf_counter()
    model_lgbm.fit(X_train, y_train)
    train_time_lgbm = time.perf_counter() - t0_lgbm

    proba_lgbm = model_lgbm.predict_proba(X_test)
    if thresholds is not None:
        idx_lgbm = {cls: i for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = []
        for p in proba_lgbm:
            if p[idx_lgbm[2]] >= thresholds[2]:
                y_pred_lgbm.append(2)
            elif p[idx_lgbm[1]] >= thresholds[1]:
                y_pred_lgbm.append(1)
            else:
                y_pred_lgbm.append(0)
        y_pred_lgbm = np.array(y_pred_lgbm)
    else:
        idx_to_cls_lgbm = {i: cls for i, cls in enumerate(model_lgbm.classes_)}
        y_pred_lgbm = np.array([idx_to_cls_lgbm[np.argmax(p)] for p in proba_lgbm])

    rec_lgbm  = recall_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    prec_lgbm = precision_score(y_test, y_pred_lgbm, average=None, labels=CLASSES, zero_division=0)
    cm_lgbm   = confusion_matrix(y_test, y_pred_lgbm, labels=CLASSES, normalize="true")

    durations_lgbm = []
    for _ in range(10):
        t0 = time.perf_counter()
        model_lgbm.predict_proba(X_test)
        durations_lgbm.append(time.perf_counter() - t0)
    inference_median_lgbm     = float(np.median(durations_lgbm))
    inference_per_sample_lgbm = inference_median_lgbm / X_test.shape[0] * 1000

    tracemalloc.start()
    model_lgbm.predict_proba(X_test)
    _, peak_lgbm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_lgbm = peak_lgbm / 1024 ** 2

    cv_lgbm = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_recalls2_lgbm = []
    cv_prec0_lgbm    = []
    for tr_idx, val_idx in cv_lgbm.split(X_train, y_train):
        X_cv_tr_lgbm  = X_train[tr_idx]
        X_cv_val_lgbm = X_train[val_idx]
        y_cv_tr_lgbm  = y_train.iloc[tr_idx]
        y_cv_val_lgbm = y_train.iloc[val_idx]
        m_clone_lgbm  = clone(model_lgbm)
        m_clone_lgbm.fit(X_cv_tr_lgbm, y_cv_tr_lgbm)
        fold_proba_lgbm = m_clone_lgbm.predict_proba(X_cv_val_lgbm)
        if thresholds is not None:
            idx_c_lgbm = {cls: i for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = []
            for p in fold_proba_lgbm:
                if p[idx_c_lgbm[2]] >= thresholds[2]:
                    fold_pred_lgbm.append(2)
                elif p[idx_c_lgbm[1]] >= thresholds[1]:
                    fold_pred_lgbm.append(1)
                else:
                    fold_pred_lgbm.append(0)
            fold_pred_lgbm = np.array(fold_pred_lgbm)
        else:
            idx_to_c_lgbm  = {i: cls for i, cls in enumerate(m_clone_lgbm.classes_)}
            fold_pred_lgbm = np.array([idx_to_c_lgbm[np.argmax(p)] for p in fold_proba_lgbm])
        cv_recalls2_lgbm.append(
            recall_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[2], average=None, zero_division=0)[0]
        )
        cv_prec0_lgbm.append(
            precision_score(y_cv_val_lgbm, fold_pred_lgbm, labels=[0], average=None, zero_division=0)[0]
        )

    results.append({
        "model":                        model_lgbm,
        "model_name":                   "LightGBM",
        "recall_class2":                float(rec_lgbm[2]),
        "recall_class1":                float(rec_lgbm[1]),
        "precision_class0":             float(prec_lgbm[0]),
        "train_time_s":                 round(train_time_lgbm, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_lgbm, 4),
        "ram_peak_mb":                  round(ram_lgbm, 2),
        "cv_recall2_mean":              round(float(np.mean(cv_recalls2_lgbm)), 4),
        "cv_recall2_std":               round(float(np.std(cv_recalls2_lgbm)), 4),
        "cv_precision0_mean":           round(float(np.mean(cv_prec0_lgbm)), 4),
        "cv_precision0_std":            round(float(np.std(cv_prec0_lgbm)), 4),
        "confusion_matrix":             cm_lgbm.tolist(),
        "history":                      None,
    })
    print(
        f"  recall_cl2={rec_lgbm[2]:.3f}  recall_cl1={rec_lgbm[1]:.3f}  "
        f"prec_cl0={prec_lgbm[0]:.3f}  train={train_time_lgbm:.2f}s  "
        f"inférence={inference_per_sample_lgbm:.4f}ms/éch  ram={ram_lgbm:.2f}MB  "
        f"CV_recall2={np.mean(cv_recalls2_lgbm):.3f}±{np.std(cv_recalls2_lgbm):.3f}  "
        f"CV_prec0={np.mean(cv_prec0_lgbm):.3f}±{np.std(cv_prec0_lgbm):.3f}"
    )

    # =========================================================
    # MODÈLE 5 — NEURAL NETWORK (Keras/TensorFlow)
    # =========================================================
    print("[NeuralNetwork] Entraînement...")

    X_tr_dense = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)
    X_te_dense = X_test.toarray()  if hasattr(X_test,  "toarray") else np.asarray(X_test)

    # Hyperparamètres :
    nn = keras.Sequential([
        keras.layers.Dense(256, activation="relu", input_shape=(X_tr_dense.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(3, activation="softmax"),
    ])
    nn.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    t0_nn = time.perf_counter()
    # Hyperparamètres :
    history_nn = nn.fit(
        X_tr_dense, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weight,
        callbacks=[EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )],
        verbose=0,
    )
    train_time_nn = time.perf_counter() - t0_nn

    proba_nn = nn.predict(X_te_dense, verbose=0)
    if thresholds is not None:
        y_pred_nn = []
        for p in proba_nn:
            if p[2] >= thresholds[2]:
                y_pred_nn.append(2)
            elif p[1] >= thresholds[1]:
                y_pred_nn.append(1)
            else:
                y_pred_nn.append(0)
        y_pred_nn = np.array(y_pred_nn)
    else:
        y_pred_nn = np.array([int(np.argmax(p)) for p in proba_nn])

    rec_nn  = recall_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    prec_nn = precision_score(y_test, y_pred_nn, average=None, labels=CLASSES, zero_division=0)
    cm_nn   = confusion_matrix(y_test, y_pred_nn, labels=CLASSES, normalize="true")

    durations_nn = []
    for _ in range(10):
        t0 = time.perf_counter()
        nn.predict(X_te_dense, verbose=0)
        durations_nn.append(time.perf_counter() - t0)
    inference_median_nn     = float(np.median(durations_nn))
    inference_per_sample_nn = inference_median_nn / X_test.shape[0] * 1000

    tracemalloc.start()
    nn.predict(X_te_dense, verbose=0)
    _, peak_nn = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_nn = peak_nn / 1024 ** 2

    results.append({
        "model":                        nn,
        "model_name":                   "NeuralNetwork",
        "recall_class2":                float(rec_nn[2]),
        "recall_class1":                float(rec_nn[1]),
        "precision_class0":             float(prec_nn[0]),
        "train_time_s":                 round(train_time_nn, 4),
        "inference_time_ms_per_sample": round(inference_per_sample_nn, 4),
        "ram_peak_mb":                  round(ram_nn, 2),
        "cv_recall2_mean":              None,
        "cv_recall2_std":               None,
        "cv_precision0_mean":           None,
        "cv_precision0_std":            None,
        "confusion_matrix":             cm_nn.tolist(),
        "history":                      history_nn.history,
    })
    print(
        f"  recall_cl2={rec_nn[2]:.3f}  recall_cl1={rec_nn[1]:.3f}  "
        f"prec_cl0={prec_nn[0]:.3f}  train={train_time_nn:.2f}s  "
        f"inférence={inference_per_sample_nn:.4f}ms/éch  ram={ram_nn:.2f}MB  "
        f"CV_recall2=N/A (trop coûteux)"
    )

    return results


def comparer_et_sauvegarder(results_avec, results_sans, artifacts_dir, models_dir):
    """
    Compare les résultats de tous les scénarios, génère des figures et sauvegarde modèles.

    Paramètres
    ----------
    results_avec : dict {scenario_name: [list of result dicts]}
        Résultats avec pénalisation asymétrique (class_weight + seuils abaissés).
    results_sans : dict {scenario_name: [list of result dicts]}
        Résultats sans pénalisation (argmax standard — baseline).
    artifacts_dir : str
        Dossier de sauvegarde des figures PNG.
    models_dir : str
        Dossier de sauvegarde des modèles (.pkl / .keras).

    Figures générées
    ----------------
    compare_recall_tres_urgent.png   → recall classe 2, avec/sans pénalisation
    compare_recall_urgent.png        → recall classe 1
    compare_precision_pas_urgent.png → précision classe 0
    compare_metriques.png            → les 3 métriques en une seule figure (6 subplots)
    compare_cv.png                   → CV recall2 et CV précision0 avec barres d'erreur
    compare_ressources.png           → train_time, inference_time, RAM
    confusions_<scenario>.png        → matrices de confusion avec/sans pénalisation par scénario
    confusions_global.png            → grille globale (tous scénarios, avec pénalisation)
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir,    exist_ok=True)

    scenarios    = list(results_avec.keys())
    model_names  = [r["model_name"] for r in list(results_avec.values())[0]]
    # Noms courts pour les axes X
    short_names  = {
        "LogisticRegression": "LR",
        "RandomForest":       "RF",
        "XGBoost":            "XGB",
        "LightGBM":           "LGBM",
        "NeuralNetwork":      "NN",
    }
    # Noms des classes en langage naturel
    class_labels = ["Pas urgent", "Urgent", "Très urgent"]
    # Couleurs par scénario : S1 bleu, S2 vert, S3 orange, S4 violet
    sc_colors    = ["#1565c0", "#2e7d32", "#e65100", "#6a1b9a"]

    x_labels  = [short_names.get(n, n) for n in model_names]
    n_models  = len(model_names)
    n_sc      = len(scenarios)

    # ==========================================================
    # FIGURES A — 3 figures séparées, une par métrique médicale
    # ==========================================================
    # Chaque figure : 2 colonnes (avec pénalisation | sans pénalisation)
    # X = modèle, barres groupées par scénario, valeur annotée sur chaque barre
    metrics_cfg = [
        ("recall_class2",    "Recall — Très urgent (classe 2)",   "compare_recall_tres_urgent.png",   "#c62828"),
        ("recall_class1",    "Recall — Urgent (classe 1)",         "compare_recall_urgent.png",         "#ef6c00"),
        ("precision_class0", "Précision — Pas urgent (classe 0)", "compare_precision_pas_urgent.png",  "#1565c0"),
    ]
    for metric_key, metric_title, fname, _ in metrics_cfg:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        fig.suptitle(metric_title, fontsize=13, fontweight="bold")
        for ax, (results, config_label) in zip(
            axes,
            [(results_avec, "Avec pénalisation"), (results_sans, "Sans pénalisation")]
        ):
            x = np.arange(n_models)
            w = 0.8 / n_sc
            for i, (sc_name, sc_results) in enumerate(results.items()):
                vals = [r[metric_key] for r in sc_results]
                bars = ax.bar(
                    x + i * w - 0.4 + w / 2, vals, width=w,
                    label=sc_name, color=sc_colors[i], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    )
            # Seuil cible 0.80 en pointillés pour repère visuel
            ax.axhline(0.80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Seuil 0.80")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.set_title(config_label, fontsize=11)
            ax.set_ylabel("Score")
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(artifacts_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Sauvegardé : {fname}")

    # ==========================================================
    # FIGURE B — 1 figure combinée (3 métriques × 2 configs = 6 subplots)
    # ==========================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Métriques médicales — tous scénarios", fontsize=13, fontweight="bold")
    for row, (metric_key, metric_title, _, _) in enumerate(metrics_cfg):
        for col, (results, config_label) in enumerate(
            [(results_avec, "Avec pénalisation"), (results_sans, "Sans pénalisation")]
        ):
            ax = axes[row][col]
            x  = np.arange(n_models)
            w  = 0.8 / n_sc
            for i, (sc_name, sc_results) in enumerate(results.items()):
                vals = [r[metric_key] for r in sc_results]
                bars = ax.bar(
                    x + i * w - 0.4 + w / 2, vals, width=w,
                    label=sc_name, color=sc_colors[i], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                )
                for bar, v in zip(bars, vals):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6,
                    )
            ax.axhline(0.80, color="red", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_title(f"{metric_title} — {config_label}", fontsize=9)
            ax.set_ylabel("Score")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(artifacts_dir, "compare_metriques.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé : compare_metriques.png")

    # ==========================================================
    # FIGURE C — CV avec barres d'erreur (±std, 5 folds)
    # ==========================================================
    # Seuls les modèles sklearn ont une CV — le NN affiche 0 avec une note
    cv_cfg = [
        ("cv_recall2_mean",    "cv_recall2_std",    "CV Recall — Très urgent (classe 2)"),
        ("cv_precision0_mean", "cv_precision0_std", "CV Précision — Pas urgent (classe 0)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Stabilité cross-validation (5 folds) — avec pénalisation", fontsize=13, fontweight="bold")
    for ax, (mean_key, std_key, cv_title) in zip(axes, cv_cfg):
        x = np.arange(n_models)
        w = 0.8 / n_sc
        for i, (sc_name, sc_results) in enumerate(results_avec.items()):
            means = [r[mean_key] if r[mean_key] is not None else 0.0 for r in sc_results]
            stds  = [r[std_key]  if r[std_key]  is not None else 0.0 for r in sc_results]
            bars  = ax.bar(
                x + i * w - 0.4 + w / 2, means, width=w,
                label=sc_name, color=sc_colors[i], alpha=0.85,
                edgecolor="white", linewidth=0.5,
                yerr=stds, capsize=3, error_kw={"elinewidth": 1, "ecolor": "black"},
            )
            for bar, v, s in zip(bars, means, stds):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + s + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                    )
        ax.axhline(0.80, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Seuil 0.80")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0, 1.20)
        ax.set_title(cv_title, fontsize=11)
        ax.set_ylabel("Score moyen (5 folds)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # Note visuelle : la barre NN vaut 0 car CV non calculée
        ax.text(n_models - 1, 0.03, "CV\nnon calculée", ha="center", fontsize=7, color="grey")
    plt.tight_layout()
    fig.savefig(os.path.join(artifacts_dir, "compare_cv.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé : compare_cv.png")

    # ==========================================================
    # FIGURE D — Ressources (temps entraînement, latence, RAM)
    # ==========================================================
    ressources_cfg = [
        ("train_time_s",                 "Temps d'entraînement (s)"),
        ("inference_time_ms_per_sample", "Latence inférence (ms / échantillon)"),
        ("ram_peak_mb",                  "RAM pic inférence (MB)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Ressources — avec pénalisation", fontsize=13, fontweight="bold")
    for ax, (res_key, res_title) in zip(axes, ressources_cfg):
        x = np.arange(n_models)
        w = 0.8 / n_sc
        for i, (sc_name, sc_results) in enumerate(results_avec.items()):
            vals = [r[res_key] if r[res_key] is not None else 0.0 for r in sc_results]
            bars = ax.bar(
                x + i * w - 0.4 + w / 2, vals, width=w,
                label=sc_name, color=sc_colors[i], alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=6, rotation=45,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(res_title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(artifacts_dir, "compare_ressources.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé : compare_ressources.png")

    # ==========================================================
    # FIGURES E — 1 figure par scénario (avec ET sans pénalisation)
    # ==========================================================
    # Chaque figure : 2 lignes (avec | sans) × 5 colonnes (modèles)
    for sc_name in scenarios:
        sc_results_avec = results_avec[sc_name]
        sc_results_sans = results_sans[sc_name]
        n_mod_sc = len(sc_results_avec)

        fig, axes = plt.subplots(2, n_mod_sc, figsize=(4 * n_mod_sc, 8))
        fig.suptitle(f"Matrices de confusion — {sc_name}", fontsize=13, fontweight="bold")

        for config_row, (config_results, config_label) in enumerate([
            (sc_results_avec, "Avec pénalisation"),
            (sc_results_sans, "Sans pénalisation"),
        ]):
            for col, r in enumerate(config_results):
                ax = axes[config_row][col]
                cm = np.array(r["confusion_matrix"])
                ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks([0, 1, 2])
                ax.set_yticks([0, 1, 2])
                ax.set_xticklabels(["Pas\nurgent", "Urgent", "Très\nurgent"], fontsize=8)
                ax.set_yticklabels(["Pas\nurgent", "Urgent", "Très\nurgent"], fontsize=8)
                for i in range(3):
                    for j in range(3):
                        ax.text(
                            j, i, f"{cm[i, j]:.2f}",
                            ha="center", va="center", fontsize=9, fontweight="bold",
                            color="white" if cm[i, j] > 0.5 else "black",
                        )
                if col == 0:
                    ax.set_ylabel(config_label, fontsize=9)
                if config_row == 1:
                    ax.set_xlabel("Prédit", fontsize=8)
                ax.set_title(short_names.get(r["model_name"], r["model_name"]), fontsize=10)

        plt.tight_layout()
        safe = sc_name.replace(" - ", "_").replace(" ", "_")
        fname = f"confusions_{safe}.png"
        fig.savefig(os.path.join(artifacts_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Sauvegardé : {fname}")

    # ==========================================================
    # FIGURE F — Grille globale (tous scénarios × tous modèles, avec pénalisation)
    # ==========================================================
    # Légende axes : PU = Pas urgent, U = Urgent, TU = Très urgent
    n_sc_f   = len(scenarios)
    n_mod_f  = n_models
    fig, axes = plt.subplots(n_sc_f, n_mod_f, figsize=(4 * n_mod_f, 4 * n_sc_f))
    fig.suptitle(
        "Matrices de confusion — vue globale (avec pénalisation)\n"
        "PU = Pas urgent  |  U = Urgent  |  TU = Très urgent",
        fontsize=12, fontweight="bold",
    )
    for row, (sc_name, sc_results) in enumerate(results_avec.items()):
        for col, r in enumerate(sc_results):
            ax = axes[row][col]
            cm = np.array(r["confusion_matrix"])
            ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_xticklabels(["PU", "U", "TU"], fontsize=7)
            ax.set_yticklabels(["PU", "U", "TU"], fontsize=7)
            for i in range(3):
                for j in range(3):
                    ax.text(
                        j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "black",
                    )
            if row == 0:
                ax.set_title(short_names.get(r["model_name"], r["model_name"]), fontsize=10)
            if col == 0:
                ax.set_ylabel(sc_name.split(" - ")[0], fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(artifacts_dir, "confusions_global.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé : confusions_global.png")

    # ==========================================================
    # SAUVEGARDE DES MODÈLES (avec pénalisation uniquement)
    # ==========================================================
    # Pipeline de déploiement S2 : préprocesseur + TF-IDF + RandomForest sauvegardés séparément
    # Le RandomForest éthique avec pénalisation est le modèle choisi pour la production
    s2_rf = next(r for r in results_avec["S2 - Ethique"] if r["model_name"] == "RandomForest")
    joblib.dump(s2_rf["preprocessor"], os.path.join(models_dir, "S2_preprocessor.pkl"))
    joblib.dump(s2_rf["tfidf"],        os.path.join(models_dir, "S2_tfidf.pkl"))
    joblib.dump(s2_rf["model"],        os.path.join(models_dir, "S2_RandomForest.pkl"))
    print("  Pipeline S2 (preprocessor + tfidf + RandomForest) sauvegardé.")

    # On sauvegarde uniquement les modèles "avec pénalisation" — ce sont ceux déployés en production
    for sc_name, sc_results in results_avec.items():
        sc_id = sc_name.split()[0]  # "S1", "S2", "S3", "S4"
        for r in sc_results:
            model      = r["model"]
            model_name = r["model_name"]
            if model_name == "NeuralNetwork":
                # Keras utilise son propre format .keras — joblib ne gère pas les poids TF
                path = os.path.join(models_dir, f"{sc_id}_{model_name}.keras")
                model.save(path)
            else:
                # joblib est plus rapide et plus fiable que pickle pour les modèles sklearn
                path = os.path.join(models_dir, f"{sc_id}_{model_name}.pkl")
                joblib.dump(model, path)
            print(f"  Modèle sauvegardé : {os.path.basename(path)}")

    print("  Modèles sauvegardés.")

    # ==========================================================
    # TRACKING MLFLOW — 1 expérience par scénario, 1 run par modèle
    # ==========================================================
    # MLflow 3.x refuse le backend fichier sans cette variable
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    # Chemin absolu converti en URI file:/// — nécessaire sur Windows
    _mlruns_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mlruns")
    )
    mlflow.set_tracking_uri(f"file:///{_mlruns_path}")

    for sc_name in scenarios:
        mlflow.set_experiment(sc_name)

        # --- runs avec pénalisation ---
        for r in results_avec[sc_name]:
            with mlflow.start_run(run_name=f"{r['model_name']} — avec pénalisation"):
                mlflow.log_params({
                    "scenario":         sc_name,
                    "model":            r["model_name"],
                    "penalize":         True,
                    "class_weight_0":   CLASS_WEIGHT[0],
                    "class_weight_1":   CLASS_WEIGHT[1],
                    "class_weight_2":   CLASS_WEIGHT[2],
                    "threshold_vital":  THRESHOLDS[2],
                    "threshold_urgent": THRESHOLDS[1],
                })
                mlflow.log_metrics({
                    "recall_class2":                r["recall_class2"],
                    "recall_class1":                r["recall_class1"],
                    "precision_class0":             r["precision_class0"],
                    "train_time_s":                 r["train_time_s"],
                    "inference_time_ms_per_sample": r["inference_time_ms_per_sample"],
                    "ram_peak_mb":                  r["ram_peak_mb"],
                })
                if r["cv_recall2_mean"] is not None:
                    mlflow.log_metrics({
                        "cv_recall2_mean":    r["cv_recall2_mean"],
                        "cv_recall2_std":     r["cv_recall2_std"],
                        "cv_precision0_mean": r["cv_precision0_mean"],
                        "cv_precision0_std":  r["cv_precision0_std"],
                    })
                # Courbe de loss epoch par epoch — uniquement disponible pour le NN
                if r.get("history") is not None:
                    for epoch, v in enumerate(r["history"].get("loss", [])):
                        mlflow.log_metric("train_loss", v, step=epoch)
                    for epoch, v in enumerate(r["history"].get("val_loss", [])):
                        mlflow.log_metric("val_loss", v, step=epoch)

        # --- runs sans pénalisation (baseline) ---
        for r in results_sans[sc_name]:
            with mlflow.start_run(run_name=f"{r['model_name']} — sans pénalisation"):
                mlflow.log_params({
                    "scenario": sc_name,
                    "model":    r["model_name"],
                    "penalize": False,
                })
                mlflow.log_metrics({
                    "recall_class2":                r["recall_class2"],
                    "recall_class1":                r["recall_class1"],
                    "precision_class0":             r["precision_class0"],
                    "train_time_s":                 r["train_time_s"],
                    "inference_time_ms_per_sample": r["inference_time_ms_per_sample"],
                    "ram_peak_mb":                  r["ram_peak_mb"],
                })
                if r["cv_recall2_mean"] is not None:
                    mlflow.log_metrics({
                        "cv_recall2_mean":    r["cv_recall2_mean"],
                        "cv_recall2_std":     r["cv_recall2_std"],
                        "cv_precision0_mean": r["cv_precision0_mean"],
                        "cv_precision0_std":  r["cv_precision0_std"],
                    })
                if r.get("history") is not None:
                    for epoch, v in enumerate(r["history"].get("loss", [])):
                        mlflow.log_metric("train_loss", v, step=epoch)
                    for epoch, v in enumerate(r["history"].get("val_loss", [])):
                        mlflow.log_metric("val_loss", v, step=epoch)

    print("  MLflow : tous les runs loggués.")
    print("\nTout sauvegardé.")


def _print_resume(resultats):
    for r in resultats:
        cv_r2_str = (
            f"{r['cv_recall2_mean']:.3f}±{r['cv_recall2_std']:.3f}"
            if r["cv_recall2_mean"] is not None else "N/A"
        )
        cv_p0_str = (
            f"{r['cv_precision0_mean']:.3f}±{r['cv_precision0_std']:.3f}"
            if r["cv_precision0_mean"] is not None else "N/A"
        )
        print(
            f"{r['model_name']:20s}  "
            f"recall_cl2={r['recall_class2']:.3f}  "
            f"recall_cl1={r['recall_class1']:.3f}  "
            f"prec_cl0={r['precision_class0']:.3f}  "
            f"train={r['train_time_s']:.2f}s  "
            f"inf={r['inference_time_ms_per_sample']:.4f}ms  "
            f"ram={r['ram_peak_mb']:.1f}MB  "
            f"CV_r2={cv_r2_str}  CV_p0={cv_p0_str}"
        )


if __name__ == "__main__":
    data_path     = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "dataset_telemed.csv")
    )
    artifacts_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "artifacts")
    )
    models_dir    = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "models")
    )

    # --- Avec pénalisation ---
    print("=== S1 - Complet (avec pénalisation) ===")
    s1_avec = run_scenario_complet(data_path, penalize=True)
    print("\n=== S2 - Ethique (avec pénalisation) ===")
    s2_avec = run_scenario_ethique(data_path, penalize=True)
    print("\n=== S3 - NLP (avec pénalisation) ===")
    s3_avec = run_scenario_nlp(data_path, penalize=True)
    print("\n=== S4 - Clinique (avec pénalisation) ===")
    s4_avec = run_scenario_clinique(data_path, penalize=True)

    # --- Sans pénalisation (baseline de comparaison) ---
    print("\n=== S1 - Complet (sans pénalisation) ===")
    s1_sans = run_scenario_complet(data_path, penalize=False)
    print("\n=== S2 - Ethique (sans pénalisation) ===")
    s2_sans = run_scenario_ethique(data_path, penalize=False)
    print("\n=== S3 - NLP (sans pénalisation) ===")
    s3_sans = run_scenario_nlp(data_path, penalize=False)
    print("\n=== S4 - Clinique (sans pénalisation) ===")
    s4_sans = run_scenario_clinique(data_path, penalize=False)

    results_avec = {
        "S1 - Complet":  s1_avec,
        "S2 - Ethique":  s2_avec,
        "S3 - NLP":      s3_avec,
        "S4 - Clinique": s4_avec,
    }
    results_sans = {
        "S1 - Complet":  s1_sans,
        "S2 - Ethique":  s2_sans,
        "S3 - NLP":      s3_sans,
        "S4 - Clinique": s4_sans,
    }

    # --- Résumés texte ---
    for sc_name, sc_results in results_avec.items():
        print(f"\n=== RÉSUMÉ {sc_name} (avec pénalisation) ===")
        _print_resume(sc_results)

    # --- Figures + sauvegarde modèles ---
    print("\n=== Génération des figures et sauvegarde ===")
    comparer_et_sauvegarder(results_avec, results_sans, artifacts_dir, models_dir)
