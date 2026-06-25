import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(_BASE_DIR, "..", "artifacts")
DATA_PATH = os.path.join(_BASE_DIR, "..", "..", "dataset_telemed.csv")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _save(filename):
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename), bbox_inches="tight", dpi=150)
    plt.close()


def load_data(path):
    df = pd.read_csv(path)
    return df


def analyse_missing_values(df):
    msno.matrix(df)
    plt.title("Matrice des valeurs manquantes")
    _save("missing_matrix.png")

    msno.bar(df)
    plt.title("Nombre de valeurs manquantes")
    _save("missing_bar.png")

    msno.heatmap(df)
    
    plt.title("Corrélation des valeurs manquantes")
    _save("missing_heatmap.png")


def analyse_distributions(df):
    colonnes = df.select_dtypes(include='number').columns

    _, axes = plt.subplots(len(colonnes), 1, figsize=(8, 4 * len(colonnes)))

    if len(colonnes) == 1:
        axes = [axes]

    for ax, col in zip(axes, colonnes):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution de {col}")

    plt.tight_layout()
    _save("distributions.png")


def analyse_correlations(df):
    df_num = df.select_dtypes(include='number')

    plt.figure(figsize=(12, 8))
    sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm")
    plt.title("Matrice de corrélation (numériques)")
    _save("correlations.png")


def analyse_boxplots(df, to_drop=None):
    df_num = df.select_dtypes(include='number').copy()
    if to_drop is not None:
        df_num = df_num.drop(columns=to_drop, errors="ignore")

    plt.figure(figsize=(12, 6))
    df_num.boxplot()
    plt.title("Boxplots des variables numériques")
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save("boxplots.png")


def analyse_categorical_distributions(df, to_drop=None):
    df = df.copy()
    if to_drop is not None:
        df = df.drop(columns=to_drop, errors="ignore")

    df_cat = df.select_dtypes(include=['object', 'category', 'bool'])

    n_cols = 3
    n_rows = (len(df_cat.columns) + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 5 * n_rows))

    for i, col in enumerate(df_cat.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(data=df, x=col)
        plt.title(f"Distribution de {col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    _save("categorical_distributions.png")


df = pd.read_csv("../../dataset_telemed.csv")
analyse_missing_values(df)
analyse_distributions(df)
analyse_correlations(df)
analyse_boxplots(df, ['niveau_urgence'])
analyse_categorical_distributions(df, to_drop=['niveau_urgence', 'patient_id', 'description_symptomes'])
