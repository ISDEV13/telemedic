# =============================================================================
# evaluate.py
# Fonctions d'évaluation et de profiling des modèles ML du projet Telemedic.
# Tous les artifacts PNG sont rangés dans artifacts/{nom_du_modele}/.
# Les tableaux et graphiques comparatifs sont sauvegardés dans artifacts/.
# =============================================================================

import os
import re
import time
import tracemalloc
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif : génère les PNG sans GUI ni tkinter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# psutil est optionnel : mesure la RAM et le CPU pendant le profiling
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# Chemin absolu vers le dossier artifacts/ (remonte d'un niveau depuis modules/)
_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

import pandas as pd
# Chemin absolu vers le dataset (remonte de 2 niveaux depuis modules/ → racine du projet)
_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset_telemed.csv")
df = pd.read_csv(_DATASET_PATH)

# Voir les valeurs typiques pour chaque classe
print(df.groupby("niveau_urgence")[["freq_cardiaque", "temp", "sat_oxygene", "tension_sys"]].mean())

def _get_model_dir(model_name, output_dir=None):
    """
    Crée et retourne le sous-dossier {output_dir}/{model_name}/.
    output_dir permet de cibler un dossier de configuration (avec/sans pénalisation).
    Si output_dir est None, utilise _ARTIFACTS_DIR par défaut.
    """
    base = output_dir if output_dir else _ARTIFACTS_DIR
    slug = model_name.replace(" ", "_")
    model_dir = os.path.join(base, slug)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir, slug


def _penalisation_label(penalized):
    """Retourne un texte court indiquant si la pénalisation est active."""
    return "Pénalisation classe vitale : OUI" if penalized else "Pénalisation classe vitale : NON"


def _weights_label(class_weights, label_names=None):
    """
    Construit une chaîne lisible des poids appliqués par classe.
    Ex: 'Pas urgent=1 · Urgent=3 · Très urgent=10'
    """
    if not class_weights:
        return "Aucune pondération"
    parts = []
    for cls, w in sorted(class_weights.items()):
        name = label_names.get(cls, str(cls)) if label_names else str(cls)
        parts.append(f"{name}={w}")
    return "Poids : " + " · ".join(parts)


# =============================================================================
# ANALYSE EXPLORATOIRE (EDA)
# =============================================================================

# Étiquettes lisibles par défaut pour la cible niveau_urgence
_LABELS_URGENCE = {0: "Pas urgent", 1: "Urgent", 2: "Très urgent"}

# Mots-outils français à ignorer dans l'analyse de fréquence du texte
# (ils n'apportent aucun signal discriminant)
_STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "au", "aux",
    "en", "dans", "sur", "pour", "par", "avec", "sans", "ce", "cet", "cette",
    "ces", "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "se", "sa", "son", "ses", "mes", "mon", "ma", "est", "pas", "ne", "que",
    "qui", "plus", "mais", "ses", "leur", "leurs",
}


def analyse_features_vs_cible(df, output_dir=None, label_names=None):
    """
    EDA bivariée — point 2 : distribution de chaque constante vitale SELON le niveau
    d'urgence. Un boxplot par feature, une boîte par classe. Révèle le "profil-type"
    de chaque niveau et quelles variables discriminent vraiment les classes.

    Args:
        df          : DataFrame contenant les features + la colonne niveau_urgence
        output_dir  : dossier de sortie (par défaut artifacts/)
        label_names : dict {classe: nom lisible}, ex {0: "Pas urgent", ...}

    Sauvegarde : eda_features_vs_cible.png
    """
    base = output_dir if output_dir else _ARTIFACTS_DIR
    os.makedirs(base, exist_ok=True)
    label_names = label_names if label_names else _LABELS_URGENCE

    # Constantes vitales numériques cliniquement parlantes — on ne garde que celles présentes
    features = ["freq_cardiaque", "tension_sys", "temp", "sat_oxygene", "age", "duree_symptomes"]
    features = [c for c in features if c in df.columns]

    # Classes triées + étiquettes lisibles pour l'axe X
    classes = sorted(df["niveau_urgence"].dropna().unique())
    xticklabels = [label_names.get(c, str(c)) for c in classes]

    # Grille de subplots, 3 colonnes
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)  # aplatit la grille pour itérer simplement

    for ax, feat in zip(axes, features):
        # Une boîte par classe : on regroupe les valeurs de la feature par niveau d'urgence
        data_par_classe = [df.loc[df["niveau_urgence"] == c, feat].dropna() for c in classes]
        ax.boxplot(data_par_classe)
        ax.set_xticks(range(1, len(classes) + 1))
        ax.set_xticklabels(xticklabels, rotation=15)
        ax.set_title(f"{feat} selon l'urgence")
        ax.set_ylabel(feat)
        ax.grid(axis="y", alpha=0.3)

    # On masque les cases vides éventuelles de la grille
    for ax in axes[len(features):]:
        ax.axis("off")

    fig.suptitle("EDA bivariée — constantes vitales × niveau d'urgence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(base, "eda_features_vs_cible.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé : eda_features_vs_cible.png")


def analyse_texte(df, output_dir=None, label_names=None, top_n=12, col="description_symptomes"):
    """
    EDA du texte libre — point 3 : analyse de la colonne description_symptomes,
    la modalité qui pilote le modèle. Produit deux vues :
      1. Longueur des descriptions (en mots) selon le niveau d'urgence
      2. Mots les plus fréquents par niveau d'urgence (hors mots-outils)

    Args:
        df          : DataFrame avec la colonne texte + niveau_urgence
        output_dir  : dossier de sortie (par défaut artifacts/)
        label_names : dict {classe: nom lisible}
        top_n       : nombre de mots les plus fréquents à afficher par classe
        col         : nom de la colonne texte

    Sauvegarde : eda_texte.png
    """
    base = output_dir if output_dir else _ARTIFACTS_DIR
    os.makedirs(base, exist_ok=True)
    label_names = label_names if label_names else _LABELS_URGENCE
    classes = sorted(df["niveau_urgence"].dropna().unique())

    # Texte en chaînes (les NaN deviennent des chaînes vides)
    textes = df[col].fillna("").astype(str)

    def tokenize(s):
        # On garde les mots de 2+ lettres (accents inclus), en minuscules, sans mots-outils
        mots = re.findall(r"\b[a-zàâäéèêëîïôöùûüç]{2,}\b", s.lower())
        return [m for m in mots if m not in _STOPWORDS_FR]

    # Longueur "brute" de chaque description (tous les mots, pour refléter la vraie taille)
    longueurs = textes.apply(lambda s: len(re.findall(r"\b\w+\b", s)))

    # Figure : 1 colonne pour la longueur + 1 colonne de top-mots par classe
    fig, axes = plt.subplots(1, len(classes) + 1, figsize=(5 * (len(classes) + 1), 5))

    # ── Vue 1 : longueur des descriptions par classe ──────────────────────────
    data_long = [longueurs[df["niveau_urgence"] == c] for c in classes]
    axes[0].boxplot(data_long)
    axes[0].set_xticks(range(1, len(classes) + 1))
    axes[0].set_xticklabels([label_names.get(c, str(c)) for c in classes], rotation=15)
    axes[0].set_title("Longueur des descriptions")
    axes[0].set_ylabel("nombre de mots")
    axes[0].grid(axis="y", alpha=0.3)

    # ── Vue 2 : top mots fréquents par classe ─────────────────────────────────
    for i, c in enumerate(classes, start=1):
        compteur = Counter()
        for s in textes[df["niveau_urgence"] == c]:
            compteur.update(tokenize(s))
        tops = compteur.most_common(top_n)
        if tops:
            mots, freqs = zip(*tops)
            y = np.arange(len(mots))
            axes[i].barh(y, freqs, color="#2e7d32")
            axes[i].set_yticks(y)
            axes[i].set_yticklabels(mots)
            axes[i].invert_yaxis()  # le mot le plus fréquent en haut
        axes[i].set_title(f"Top mots — {label_names.get(c, str(c))}")
        axes[i].set_xlabel("fréquence")

    fig.suptitle("EDA texte — longueur et vocabulaire par niveau d'urgence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(base, "eda_texte.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Sauvegardé : eda_texte.png")


# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate_model(model_name, history, y_true, y_pred, labels=None, label_names=None, penalized=False, class_weights=None, output_dir=None):
    """
    Évalue un modèle de classification et sauvegarde les résultats en PNG.

    Args:
        model_name    : nom du modèle (str), sert aussi de nom de sous-dossier
        history       : objet History Keras issu de model.fit (None pour sklearn)
        y_true        : vraies étiquettes (liste ou array)
        y_pred        : étiquettes prédites par le modèle (argmax déjà appliqué)
        labels        : liste ordonnée des classes, ex: [0, 1, 2]. Auto si None.
        label_names   : dict {classe: nom lisible}, ex: {0: "Pas urgent", ...}
        penalized     : True si le modèle a été entraîné avec pénalisation
        class_weights : dict {classe: poids}, ex: {0: 1, 1: 3, 2: 10} — affiché dans les charts

    Returns:
        dict contenant toutes les métriques + la matrice de confusion + le statut de pénalisation
    """

    # ── 1. Détermination des classes présentes ────────────────────────────────
    classes = labels if labels is not None else sorted(set(y_true))

    # Noms lisibles pour l'affichage — si label_names n'est pas fourni on utilise
    # les valeurs brutes (0, 1, 2)
    def to_display(cls):
        if label_names:
            return label_names.get(cls, str(cls))
        return str(cls)

    display_names = [to_display(cls) for cls in classes]

    # ── 2. Métriques par classe ───────────────────────────────────────────────
    # average=None → sklearn retourne un tableau, une valeur par classe
    # zero_division=0 → évite une erreur si une classe n'a aucune prédiction
    precision_par_classe = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    recall_par_classe    = recall_score   (y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_par_classe        = f1_score       (y_true, y_pred, average=None, labels=classes, zero_division=0)

    # Support = nombre de vrais exemples de chaque classe dans y_true
    support_par_classe = np.array([np.sum(np.array(y_true) == cls) for cls in classes])

    # ── 3. Construction du dictionnaire de résultats ──────────────────────────
    result = {
        "model_name":         model_name,
        "penalized":          penalized,
        "accuracy":           accuracy_score(y_true, y_pred),
        "f1_weighted":        f1_score       (y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score   (y_true, y_pred, average="weighted", zero_division=0),
        "par_classe": {
            cls: {
                "precision": float(precision_par_classe[i]),
                "recall":    float(recall_par_classe[i]),
                "f1":        float(f1_par_classe[i]),
                "support":   int(support_par_classe[i]),
            }
            for i, cls in enumerate(classes)
        },
        # Ces deux valeurs ne sont disponibles que pour les réseaux de neurones (Keras)
        "epochs":             len(history.history.get("loss", [])) if history else None,
        "val_accuracy_final": history.history["val_accuracy"][-1]  if history and "val_accuracy" in history.history else None,
    }

    # ── 4. Affichage console ──────────────────────────────────────────────────
    print("=" * 65)
    print(f"  ÉVALUATION : {model_name}  |  {_penalisation_label(penalized)}")
    print("=" * 65)
    print(f"  Accuracy : {result['accuracy']:.4f}")
    if result["val_accuracy_final"] is not None:
        print(f"  Val accuracy (ep. {result['epochs']}) : {result['val_accuracy_final']:.4f}")
    print()
    print(f"  {'Classe':<14} {'Précision':>10} {'Rappel':>10} {'F1':>10} {'Support':>10}")
    print("  " + "-" * 56)
    for cls, m in result["par_classe"].items():
        print(f"  {to_display(cls):<14} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['support']:>10}")
    print("=" * 65)

    # ── 5. Dossier de destination pour ce modèle ──────────────────────────────
    # output_dir permet de ranger les artifacts dans avec_penalisation/ ou sans_penalisation/
    model_dir, slug = _get_model_dir(model_name, output_dir)

    # ── 6. Matrice de confusion normalisée (PNG) ──────────────────────────────
    # normalize='true' → chaque cellule est un taux (0.0 à 1.0) par classe réelle
    # Plus lisible que les effectifs bruts quand les classes sont déséquilibrées
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")

    # On stocke la matrice normalisée dans le résultat pour benchmark_confusion
    result["confusion_matrix"] = cm.tolist()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(f"Matrice de confusion — {model_name}")
    # Sous-titre avec statut de pénalisation + poids par classe si disponibles
    pen_line = _penalisation_label(penalized)
    if penalized and class_weights:
        pen_line += f"\n{_weights_label(class_weights, label_names)}"
    ax.set_xlabel(ax.get_xlabel() + f"\n{pen_line}", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{slug}_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ── 7. Tableau des métriques par classe (PNG) ─────────────────────────────
    col_labels_tbl = ["Précision", "Rappel", "F1", "Support"]
    row_labels_tbl = display_names.copy()
    cell_vals = [
        [
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            str(m["support"]),
        ]
        for m in result["par_classe"].values()
    ]

    # Ligne de synthèse globale en bas du tableau
    row_labels_tbl.append("Global (wtd)")
    cell_vals.append([
        f"{result['precision_weighted']:.4f}",
        f"{result['recall_weighted']:.4f}",
        f"{result['f1_weighted']:.4f}",
        str(len(y_true)),
    ])

    fig, ax = plt.subplots(figsize=(8, 0.6 * (len(row_labels_tbl) + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_vals,
        rowLabels=row_labels_tbl,
        colLabels=col_labels_tbl,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    # Le titre contient le statut de pénalisation + les poids par classe si disponibles
    pen_line = _penalisation_label(penalized)
    if penalized and class_weights:
        pen_line += f"  |  {_weights_label(class_weights, label_names)}"
    ax.set_title(
        f"Métriques — {model_name}  (accuracy={result['accuracy']:.4f})\n{pen_line}",
        fontsize=10,
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"{slug}_metrics_table.png"), dpi=150)
    plt.close(fig)

    # ── 8. Courbe de loss — uniquement pour les réseaux de neurones ───────────
    if history is not None:
        hist = history.history
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(hist["loss"], label="Train loss")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label="Val loss")
        ax.set_xlabel("Époque")
        ax.set_ylabel("Loss")
        ax.set_title(f"Courbe de loss — {model_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, f"{slug}_loss_curve.png"), dpi=150)
        plt.close(fig)

    return result


# =============================================================================
# PROFILING
# =============================================================================

def profile_model(model_name, predict_fn, X_test, train_fn=None, n_repeat=10):
    """
    Mesure les performances techniques d'un modèle : temps et mémoire.

    Args:
        model_name : nom du modèle (str)
        predict_fn : fonction callable(X) qui retourne les prédictions
        X_test     : données de test utilisées pour mesurer l'inférence
        train_fn   : fonction callable() wrappant model.fit — None si déjà entraîné
        n_repeat   : nombre de passes d'inférence pour avoir une mesure stable

    Returns:
        dict avec les métriques de performance technique
    """
    process = psutil.Process() if _PSUTIL else None
    n_samples = X_test.shape[0]

    # ── 1. Temps et RAM pendant l'entraînement ────────────────────────────────
    # Mesuré uniquement si train_fn est fourni
    train_time_s       = None
    ram_train_delta_mb = None

    if train_fn is not None:
        ram_before         = process.memory_info().rss if process else 0
        t0                 = time.perf_counter()
        train_fn()
        train_time_s       = time.perf_counter() - t0
        ram_train_delta_mb = (process.memory_info().rss - ram_before) / 1024 ** 2 if process else None

    # ── 2. Temps d'inférence (médiane sur n_repeat passes) ────────────────────
    # La médiane est plus robuste que la moyenne face aux pics ponctuels
    durations = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        predict_fn(X_test)
        durations.append(time.perf_counter() - t0)

    inference_time_total_s = float(np.median(durations))

    # ── 3. RAM consommée au pic pendant l'inférence ───────────────────────────
    # tracemalloc trace les allocations Python — peak = valeur maximale atteinte
    tracemalloc.start()
    predict_fn(X_test)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_inference_delta_mb = peak / 1024 ** 2

    # ── 4. CPU pendant l'inférence ────────────────────────────────────────────
    # Premier appel pour reset le compteur, second appel après une inférence
    cpu_inference_percent = None
    if process:
        process.cpu_percent(interval=None)
        predict_fn(X_test)
        cpu_inference_percent = process.cpu_percent(interval=None)

    result = {
        "model_name":                   model_name,
        "train_time_s":                 round(train_time_s, 4) if train_time_s is not None else None,
        "inference_time_total_s":       round(inference_time_total_s, 6),
        "inference_time_per_sample_ms": round(inference_time_total_s / n_samples * 1000, 4),
        "ram_train_delta_mb":           round(ram_train_delta_mb, 2) if ram_train_delta_mb is not None else None,
        "ram_inference_peak_mb":        round(ram_inference_delta_mb, 2),
        "cpu_inference_percent":        round(cpu_inference_percent, 1) if cpu_inference_percent is not None else None,
    }

    # ── 5. Affichage console ──────────────────────────────────────────────────
    print("=" * 55)
    print(f"  PROFILING : {model_name}")
    print("=" * 55)
    if result["train_time_s"] is not None:
        print(f"  Entraînement           : {result['train_time_s']:.4f} s")
        if result["ram_train_delta_mb"] is not None:
            print(f"  RAM entraînement Δ     : {result['ram_train_delta_mb']:.2f} MB")
    print(f"  Inférence totale       : {result['inference_time_total_s']:.6f} s  (médiane {n_repeat} runs)")
    print(f"  Inférence / échantillon: {result['inference_time_per_sample_ms']:.4f} ms")
    print(f"  RAM inférence (peak)   : {result['ram_inference_peak_mb']:.2f} MB")
    if result["cpu_inference_percent"] is not None:
        print(f"  CPU inférence          : {result['cpu_inference_percent']:.1f} %")
    print("=" * 55)

    return result


# =============================================================================
# BENCHMARKS COMPARATIFS
# =============================================================================

def benchmark_models(results, label_names=None, output_dir=None):
    """
    Génère un tableau PNG comparant les métriques de tous les modèles côte à côte.
    Chaque ligne = un modèle. Colonnes = précision / rappel / F1 par classe + accuracy + pénalisation.
    Sauvegardé dans output_dir/benchmark_metrics.png (ou artifacts/ par défaut).
    """
    if not results:
        return

    classes = list(results[0]["par_classe"].keys())

    def display(cls):
        if label_names:
            return label_names.get(cls, str(cls))
        return str(cls)

    # ── En-têtes : une colonne P/R/F1 par classe + Accuracy + Pénalisé ────────
    col_labels = ["Modèle"]
    for cls in classes:
        name = display(cls)
        col_labels += [f"P\n{name}", f"R\n{name}", f"F1\n{name}"]
    col_labels += ["Accuracy", "Pénalisé"]

    # ── Lignes : une par modèle ────────────────────────────────────────────────
    cell_vals = []
    for r in results:
        row = [r["model_name"]]
        for cls in classes:
            m = r["par_classe"][cls]
            row += [f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}"]
        row.append(f"{r['accuracy']:.3f}")
        row.append("Oui" if r.get("penalized") else "Non")
        cell_vals.append(row)

    n_cols = len(col_labels)
    n_rows = len(cell_vals)

    fig, ax = plt.subplots(figsize=(2.0 * n_cols, 0.9 * (n_rows + 2)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_vals,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.2)
    ax.set_title("Comparaison des modèles — Métriques par classe", fontsize=13, pad=15)
    plt.tight_layout()
    base = output_dir if output_dir else _ARTIFACTS_DIR
    plt.savefig(os.path.join(base, "benchmark_metrics.png"), dpi=150)
    plt.close(fig)
    print(f"  → benchmark_metrics.png sauvegardé dans {base}")


def benchmark_metrics_chart(results, label_names=None, output_dir=None):
    """
    Génère un PNG avec 4 sous-graphiques comparant tous les modèles :
      - F1, Recall, Précision par classe (barres groupées par modèle)
      - Accuracy globale (une barre par modèle)
    Les modèles pénalisés sont marqués ★ et leurs barres sont hachurées //.
    Sauvegardé dans output_dir/benchmark_metriques.png (ou artifacts/ par défaut).
    """
    if not results:
        return

    classes   = list(results[0]["par_classe"].keys())
    n_classes = len(classes)
    n_models  = len(results)

    def display(cls):
        if label_names:
            return label_names.get(cls, str(cls))
        return str(cls)

    display_names = [display(cls) for cls in classes]

    # Nom du modèle sur l'axe X — ★ si pénalisation active
    model_labels = [
        r["model_name"] + (" ★" if r.get("penalized") else "")
        for r in results
    ]

    colors    = ["#4C72B0", "#DD8452", "#55A868"]
    bar_width = 0.2
    x         = np.arange(n_models)

    fig, axes = plt.subplots(2, 2, figsize=(max(14, 3 * n_models), 11))
    axes = axes.flatten()  # on aplatit pour itérer avec un index simple

    # ── 3 premiers sous-graphiques : F1 / Recall / Précision par classe ───────
    # Chaque sous-graphique = barres groupées (un groupe par modèle, une barre par classe)
    metrics_config = [
        ("f1",        "F1-score par classe"),
        ("recall",    "Recall par classe"),
        ("precision", "Précision par classe"),
    ]

    for ax_idx, (metric_key, title) in enumerate(metrics_config):
        ax = axes[ax_idx]

        for i, (cls, color, dname) in enumerate(zip(classes, colors, display_names)):
            vals      = [r["par_classe"][cls][metric_key] for r in results]
            positions = x + i * bar_width - (n_classes - 1) * bar_width / 2

            # On dessine barre par barre pour appliquer le hachage individuellement
            for j, (pos, val, r) in enumerate(zip(positions, vals, results)):
                hatch = "//" if r.get("penalized") else ""
                ax.bar(
                    pos, val,
                    width=bar_width,
                    color=color,
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.5,
                    label=dname if j == 0 else "",  # légende affichée une seule fois
                )

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=11)
        ax.legend(title="Classe", fontsize=8, loc="lower right")
        # Ligne de référence à 0.8 pour repérer visuellement les modèles faibles
        ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # ── 4ème sous-graphique : Accuracy globale (une barre par modèle) ─────────
    ax = axes[3]
    for j, r in enumerate(results):
        hatch = "//" if r.get("penalized") else ""
        ax.bar(j, r["accuracy"], width=0.5, color="#8172B2",
               hatch=hatch, edgecolor="black", linewidth=0.5)
        # Valeur affichée au-dessus de chaque barre
        ax.text(j, r["accuracy"] + 0.01, f"{r['accuracy']:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Accuracy globale", fontsize=11)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # Note commune en bas de figure
    if any(r.get("penalized") for r in results):
        fig.text(0.5, 0.005, "★  //  = pénalisation classe vitale active",
                 ha="center", fontsize=9, color="gray")

    fig.suptitle("Comparaison des modèles — Métriques complètes", fontsize=14, y=1.01)
    plt.tight_layout()
    base = output_dir if output_dir else _ARTIFACTS_DIR
    plt.savefig(os.path.join(base, "benchmark_metriques.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → benchmark_metriques.png sauvegardé dans {base}")


def benchmark_resources(profiles, output_dir=None):
    """
    Génère un tableau PNG comparant les performances techniques de tous les modèles.
    Chaque ligne = un modèle. Colonnes = temps, inférence, RAM, CPU.
    Sauvegardé dans output_dir/benchmark_ressources.png (ou artifacts/ par défaut).
    """
    if not profiles:
        return

    col_labels = [
        "Modèle",
        "Train (s)",
        "Inférence (s)",
        "ms / échantillon",
        "RAM pic (MB)",
        "CPU (%)",
    ]

    cell_vals = []
    for p in profiles:
        row = [
            p["model_name"],
            f"{p['train_time_s']:.3f}"          if p["train_time_s"]          is not None else "—",
            f"{p['inference_time_total_s']:.6f}",
            f"{p['inference_time_per_sample_ms']:.4f}",
            f"{p['ram_inference_peak_mb']:.2f}",
            f"{p['cpu_inference_percent']:.1f}"  if p["cpu_inference_percent"] is not None else "—",
        ]
        cell_vals.append(row)

    n_rows = len(cell_vals)

    fig, ax = plt.subplots(figsize=(14, 0.8 * (n_rows + 2)))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_vals,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.0)
    ax.set_title("Comparaison des modèles — Performances ressources", fontsize=13, pad=15)
    plt.tight_layout()
    base = output_dir if output_dir else _ARTIFACTS_DIR
    plt.savefig(os.path.join(base, "benchmark_ressources.png"), dpi=150)
    plt.close(fig)
    print(f"  → benchmark_ressources.png sauvegardé dans {base}")


def benchmark_confusion(results, label_names=None, output_dir=None):
    """
    Génère une figure PNG avec toutes les matrices de confusion côte à côte.
    Permet un coup d'œil global pour comparer les modèles visuellement.
    Le statut de pénalisation est indiqué sous le titre de chaque matrice.
    Sauvegardé dans artifacts/benchmark_confusion.png.
    """
    if not results:
        return

    classes = list(results[0]["par_classe"].keys())
    if label_names:
        display_names = [label_names.get(cls, str(cls)) for cls in classes]
    else:
        display_names = [str(cls) for cls in classes]

    n_models = len(results)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))

    # Si un seul modèle, axes est un objet unique et non une liste → on le normalise
    if n_models == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")

        # Titre = nom du modèle + statut de pénalisation sur une deuxième ligne
        pen_label = "★ Pénalisé" if r.get("penalized") else "Non pénalisé"
        ax.set_title(f"{r['model_name']}\n{pen_label}", fontsize=10, pad=8)

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), fontsize=8)

    fig.suptitle("Matrices de confusion — Vue comparative", fontsize=13, y=1.02)
    plt.tight_layout()
    base = output_dir if output_dir else _ARTIFACTS_DIR
    plt.savefig(os.path.join(base, "benchmark_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → benchmark_confusion.png sauvegardé dans {base}")


# =============================================================================
# EXÉCUTION DIRECTE — génère les figures d'analyse exploratoire
# =============================================================================
# Ce bloc ne s'exécute QUE si on lance `python modules/evaluate.py` directement,
# pas lors d'un simple import du module (évite de relancer les calculs à chaque import).
if __name__ == "__main__":
    analyse_features_vs_cible(df)
    analyse_texte(df)
