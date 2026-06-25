from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np  # utilisé par apply_business_rules pour remplacer les valeurs aberrantes par NaN
import os


# Tranches d'âge médicalement significatives (généralisation d'attribut)
_AGE_BINS   = [0, 17, 40, 64, float("inf")]
_AGE_LABELS = ["enfant", "adulte_jeune", "adulte", "senior"]
_CAT_COL = ["sexe", "zone_vie",'source']  
_TXT_COL = ['description_symptomes']
_NUM_COL = ['age', 'freq_cardiaque', 'frequence_cardiaque', 'tension_sys', 'temp', 'sat_oxygene','antecedents','duree_symptomes']  
_TRASH = ['patient_id']  # Variables à supprimer (ex: ID patient)
_TARGET = ["niveau_urgence"]  # Variable cible pour la prédiction

_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset_telemed.csv")

df = pd.read_csv(_DATA_PATH)

# Voir les valeurs typiques pour chaque classe
print(df.groupby("niveau_urgence")[["freq_cardiaque", "temp", "sat_oxygene", "tension_sys"]].mean())

# Bornes des valeurs physiquement impossibles (pas des seuils cliniques)
_BORNES_IMPOSSIBLES = {
    "age":             (0,   130),   # aucun humain au-delà de 122 ans
    "freq_cardiaque":  (0,   400),   # au-delà de 400 bpm = impossible
    "tension_sys":     (0,   400),   # limite physique des capteurs
    "temp":            (0,   60),    # impossible sur un être vivant mesurable
    "sat_oxygene":     (0,   100),   # pourcentage, bornes strictes
    "duree_symptomes": (0,   None),  # durée négative impossible
    "antecedents":     (0,   None),  # comptage négatif impossible
}


def apply_business_rules(df):
    df = df.copy()
    rapport = []

    # =========================
    # 1. Suppression des doublons
    # =========================
    n_avant = len(df)
    df = df.drop_duplicates()
    n_apres = len(df)
    n_supprime = n_avant - n_apres

    if n_supprime > 0:
        rapport.append(
            f"Doublons supprimés : {n_supprime} ligne(s) ({n_avant} → {n_apres})"
        )
        
    for col, (min_val, max_val) in _BORNES_IMPOSSIBLES.items():

        if col not in df.columns:
            continue

        masque_col = pd.Series(False, index=df.index)

        if min_val is not None:
            masque_col |= df[col] < min_val

        if max_val is not None:
            masque_col |= df[col] > max_val

        n_anomalies = masque_col.sum()

        if n_anomalies > 0:
            df.loc[masque_col, col] = np.nan
            rapport.append(
                f"{col} : {n_anomalies} valeur(s) impossible(s) → remplacées par NaN"
            )


    print("=" * 55)
    print("RAPPORT RÈGLES MÉTIERS")
    print("=" * 55)

    if rapport:
        for r in rapport:
            print(" •", r)
    else:
        print("Aucune anomalie détectée.")

    print("=" * 55)

    return df


# Variables sensibles à surveiller selon les scénarios

def ethical_preprocessing(df, sensitive_cols=None):
    """
    Généralisation des attributs sensibles avant modélisation.

    - age             : discrétisé en tranches médicales → variable catégorielle
    - sensitive_cols  : supprimées si renseignées, ignorées sinon

    Args:
        df               : DataFrame source (non modifié)
        sensitive_cols   : liste des colonnes sensibles à supprimer

    Returns:
        DataFrame transformé
    """
    df = df.copy()
    rapport = []

    # Généralisation de l'âge : valeur précise → tranche
    if "age" in df.columns:
        df["age"] = pd.cut(df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=True)
        rapport.append("age : généralisation → tranches [enfant | adulte_jeune | adulte | senior]")

    # Suppression des variables sensibles si renseignées
    if sensitive_cols is not None:
        present = [c for c in sensitive_cols if c in df.columns]
        df = df.drop(columns=present)
        rapport.append(f"Variables sensibles supprimées : {present}")
    else:
        rapport.append("Aucune variable sensible renseignée")

    print("=" * 55)
    print("    RAPPORT PREPROCESSING ÉTHIQUE")
    print("=" * 55)
    for ligne in rapport:
        print(f"  • {ligne}")
    print("=" * 55)

    return df


def preprocessingTechnique(df, target_col, to_drop=None, num_cols=None, cat_cols=None):
    df = df.copy()
    rapport = []

    if to_drop:
        dropped = [c for c in to_drop if c in df.columns]
        df = df.drop(columns=to_drop, errors="ignore")
        rapport.append(f"Colonnes supprimées ({len(dropped)}) : {dropped}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Détection automatique uniquement si non renseigné
    if num_cols is None:
        num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        rapport.append(f"Colonnes numériques détectées automatiquement ({len(num_cols)}) : {num_cols}")
    else:
        rapport.append(f"Colonnes numériques fournies ({len(num_cols)}) : {num_cols}")

    if cat_cols is None:
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        rapport.append(f"Colonnes catégorielles détectées automatiquement ({len(cat_cols)}) : {cat_cols}")
    else:
        rapport.append(f"Colonnes catégorielles fournies ({len(cat_cols)}) : {cat_cols}")

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Assemblage (uniquement les types présents)
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
        rapport.append(f"Numériques → imputation médiane + MinMaxScaler")
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
        rapport.append(f"Catégorielles → imputation mode + OneHotEncoder")

    preprocessor = ColumnTransformer(transformers)

    X_processed = preprocessor.fit_transform(X)
    
    feature_names = preprocessor.get_feature_names_out()
    X_processed = pd.DataFrame(X_processed, columns=feature_names)

    df_final = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

    # Vérification valeurs manquantes
    nan_avant = df.drop(columns=[target_col]).isnull().sum().sum()
    nan_apres = X_processed.isnull().sum().sum()
    rapport.append(f"Valeurs manquantes avant : {nan_avant} → après : {nan_apres}")

    # Dimensions
    rapport.append(f"Dimensions avant  : {df.drop(columns=[target_col]).shape}")
    rapport.append(f"Dimensions après  : {X_processed.shape}")

    # Affichage
    print("=" * 55)
    print("        RAPPORT DE PREPROCESSING")
    print("=" * 55)
    for ligne in rapport:
        print(f"  • {ligne}")
    print("=" * 55)

    return X_processed, y, preprocessor, df_final


def split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divise les données en ensembles d'entraînement et de test.

    Args:
        test_size    : proportion du jeu de test (défaut 20 %)
        random_state : graine pour la reproductibilité

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

df = pd.read_csv(_DATA_PATH)


# Suppresion des colonnes id et source non pertinante pour le modèle. On garde l'age meme si éthiquement limite car très pertinant pour le niveau d'urgence
# X_processed, y, preprocessor, df_final = preprocessingTechnique(
#     df,
#     target_col="niveau_urgence",
# )

# df_final.head()

# X_train, X_test, y_train, y_test = split(X_processed,y)