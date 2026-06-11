import json
import os
import time
import tracemalloc
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model_name, history, y_true, y_pred, labels=None):
    """
    Évalue un modèle de classification et retourne les métriques par classe.

    Args:
        model_name : nom du modèle (str)
        history    : objet History Keras issu de model.fit (peut être None)
        y_true     : vraies étiquettes
        y_pred     : étiquettes prédites (argmax déjà appliqué)
        labels     : liste ordonnée des classes (ex: [0, 1, 2]) — détection auto si None

    Returns:
        dict {
            "model_name"          : str,
            "accuracy"            : float,
            "f1_weighted"         : float,
            "precision_weighted"  : float,
            "recall_weighted"     : float,
            "par_classe"          : { classe: { precision, recall, f1 } },
            "epochs"              : int | None,
            "val_accuracy_final"  : float | None,
        }
    """
    classes = labels if labels is not None else sorted(set(y_true))

    precision_par_classe = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    recall_par_classe    = recall_score   (y_true, y_pred, average=None, labels=classes, zero_division=0)
    f1_par_classe        = f1_score       (y_true, y_pred, average=None, labels=classes, zero_division=0)

    result = {
        "model_name":          model_name,
        "accuracy":            accuracy_score(y_true, y_pred),
        "f1_weighted":         f1_score       (y_true, y_pred, average="weighted", zero_division=0),
        "precision_weighted":  precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":     recall_score   (y_true, y_pred, average="weighted", zero_division=0),
        "par_classe": {
            cls: {
                "precision": float(precision_par_classe[i]),
                "recall":    float(recall_par_classe[i]),
                "f1":        float(f1_par_classe[i]),
            }
            for i, cls in enumerate(classes)
        },
        "epochs":             len(history.history.get("loss", [])) if history else None,
        "val_accuracy_final": history.history["val_accuracy"][-1]  if history and "val_accuracy" in history.history else None,
    }

    # ── Affichage ─────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"  ÉVALUATION : {model_name}")
    print("=" * 55)
    print(f"  Accuracy          : {result['accuracy']:.4f}")
    print(f"  F1 (weighted)     : {result['f1_weighted']:.4f}")
    print(f"  Précision (wtd)   : {result['precision_weighted']:.4f}")
    print(f"  Rappel    (wtd)   : {result['recall_weighted']:.4f}")
    if result["val_accuracy_final"] is not None:
        print(f"  Val accuracy (ep. {result['epochs']}) : {result['val_accuracy_final']:.4f}")
    print()
    print(f"  {'Classe':<10} {'Précision':>10} {'Rappel':>10} {'F1':>10}")
    print("  " + "-" * 42)
    for cls, m in result["par_classe"].items():
        print(f"  {str(cls):<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("=" * 55)

    # ── Sauvegarde artifacts ───────────────────────────────────────────────────
    slug = model_name.replace(" ", "_")

    # JSON des métriques
    json_path = os.path.join(_ARTIFACTS_DIR, f"{slug}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(_ARTIFACTS_DIR, f"{slug}_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Tableau des métriques par classe
    col_labels = ["Précision", "Rappel", "F1"]
    row_labels  = [f"Classe {cls}" for cls in classes]
    cell_vals   = [
        [f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}"]
        for m in result["par_classe"].values()
    ]
    # Ligne de synthèse globale
    row_labels.append("Global (wtd)")
    cell_vals.append([
        f"{result['precision_weighted']:.4f}",
        f"{result['recall_weighted']:.4f}",
        f"{result['f1_weighted']:.4f}",
    ])

    fig, ax = plt.subplots(figsize=(7, 0.6 * (len(row_labels) + 2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_vals,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    ax.set_title(f"Métriques — {model_name}  (accuracy={result['accuracy']:.4f})",
                 fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(_ARTIFACTS_DIR, f"{slug}_metrics_table.png"), dpi=150)
    plt.close(fig)

    # Courbe de loss (uniquement si history Keras fourni)
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
        plt.savefig(os.path.join(_ARTIFACTS_DIR, f"{slug}_loss_curve.png"), dpi=150)
        plt.close(fig)

    return result


# ── Profiling ─────────────────────────────────────────────────────────────────

def profile_model(model_name, predict_fn, X_test, train_fn=None, n_repeat=10):
    """
    Mesure le temps d'inférence, le temps d'entraînement et les ressources consommées.

    Args:
        model_name : nom du modèle (str)
        predict_fn : callable(X) → prédictions
        X_test     : données de test pour l'inférence
        train_fn   : callable sans argument wrappant model.fit — None si déjà entraîné
        n_repeat   : nombre de passages pour moyenner l'inférence (défaut : 10)

    Returns:
        dict {
            "model_name"                  : str,
            "train_time_s"                : float | None,
            "inference_time_total_s"      : float,
            "inference_time_per_sample_ms": float,
            "ram_train_delta_mb"          : float | None,
            "ram_inference_delta_mb"      : float,
            "cpu_inference_percent"       : float | None,
        }
    """
    process = psutil.Process() if _PSUTIL else None
    n_samples = X_test.shape[0]

    # ── Temps + RAM entraînement ───────────────────────────────────────────────
    train_time_s = None
    ram_train_delta_mb = None
    if train_fn is not None:
        ram_before = process.memory_info().rss if process else 0
        t0 = time.perf_counter()
        train_fn()
        train_time_s = time.perf_counter() - t0
        ram_train_delta_mb = (process.memory_info().rss - ram_before) / 1024 ** 2 if process else None

    # ── Temps + RAM inférence (médiane sur n_repeat passages) ─────────────────
    durations = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        predict_fn(X_test)
        durations.append(time.perf_counter() - t0)

    ram_before = process.memory_info().rss if process else 0
    tracemalloc.start()
    predict_fn(X_test)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ram_inference_delta_mb = peak / 1024 ** 2

    inference_time_total_s = float(np.median(durations))

    # ── CPU pendant l'inférence ───────────────────────────────────────────────
    cpu_inference_percent = None
    if process:
        process.cpu_percent(interval=None)          # reset du compteur
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

    print("=" * 55)
    print(f"  PROFILING : {model_name}")
    print("=" * 55)
    if result["train_time_s"] is not None:
        print(f"  Entraînement          : {result['train_time_s']:.4f} s")
        if result["ram_train_delta_mb"] is not None:
            print(f"  RAM entraînement Δ    : {result['ram_train_delta_mb']:.2f} MB")
    print(f"  Inférence totale      : {result['inference_time_total_s']:.6f} s  (médiane {n_repeat} runs)")
    print(f"  Inférence / échantillon: {result['inference_time_per_sample_ms']:.4f} ms")
    print(f"  RAM inférence (peak)  : {result['ram_inference_peak_mb']:.2f} MB")
    if result["cpu_inference_percent"] is not None:
        print(f"  CPU inférence         : {result['cpu_inference_percent']:.1f} %")
    print("=" * 55)

    slug = model_name.replace(" ", "_")
    json_path = os.path.join(_ARTIFACTS_DIR, f"{slug}_profile.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


# ── Benchmark ─────────────────────────────────────────────────────────────────

def benchmark_models(results, profiles=None, sort_by="f1_weighted"):
    """
    Compare plusieurs modèles à partir des dicts retournés par evaluate_model.
    Intègre optionnellement les données de profile_model.

    Args:
        results  : liste de dicts (sorties de evaluate_model)
        profiles : liste de dicts (sorties de profile_model) — optionnel
        sort_by  : colonne de tri du classement (défaut : f1_weighted)

    Returns:
        DataFrame classé par sort_by décroissant
    """
    # Index des profils par nom de modèle pour jointure rapide
    profiles_by_name = {p["model_name"]: p for p in profiles} if profiles else {}

    rows = []
    for r in results:
        row = {
            "model":              r["model_name"],
            "accuracy":           r["accuracy"],
            "f1_weighted":        r["f1_weighted"],
            "precision_weighted": r["precision_weighted"],
            "recall_weighted":    r["recall_weighted"],
        }
        for cls, m in r["par_classe"].items():
            row[f"f1_classe_{cls}"]        = m["f1"]
            row[f"precision_classe_{cls}"] = m["precision"]
            row[f"recall_classe_{cls}"]    = m["recall"]

        # Fusion des données de profiling si disponibles
        p = profiles_by_name.get(r["model_name"])
        if p:
            row["train_time_s"]                 = p.get("train_time_s")
            row["inference_time_per_sample_ms"] = p.get("inference_time_per_sample_ms")
            row["ram_inference_peak_mb"]        = p.get("ram_inference_peak_mb")
            row["cpu_inference_percent"]        = p.get("cpu_inference_percent")

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)
    df.index += 1

    # ── Affichage ─────────────────────────────────────────────────────────────
    global_cols = ["model", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    print("=" * 70)
    print("  BENCHMARK — MÉTRIQUES GLOBALES")
    print("=" * 70)
    print(df[global_cols].to_string())
    print()

    f1_class_cols = ["model"] + [c for c in df.columns if c.startswith("f1_classe_")]
    print("  BENCHMARK — F1 PAR CLASSE")
    print("=" * 70)
    print(df[f1_class_cols].to_string())

    perf_cols = [c for c in ["model", "train_time_s", "inference_time_per_sample_ms",
                              "ram_inference_peak_mb", "cpu_inference_percent"] if c in df.columns]
    if len(perf_cols) > 1:
        print()
        print("  BENCHMARK — RESSOURCES")
        print("=" * 70)
        print(df[perf_cols].to_string())
    print("=" * 70)

    # ── Sauvegarde artifacts ───────────────────────────────────────────────────
    df.to_csv(os.path.join(_ARTIFACTS_DIR, "benchmark_summary.csv"), index_label="rank")

    x = np.arange(len(df))

    # Graphe métriques globales
    metrics_globales = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]
    width = 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 5))
    for i, metric in enumerate(metrics_globales):
        ax.bar(x + i * width, df[metric], width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark — métriques globales")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(_ARTIFACTS_DIR, "benchmark_global.png"), dpi=150)
    plt.close(fig)

    # Graphe F1 par classe
    f1_cols = [c for c in df.columns if c.startswith("f1_classe_")]
    width = 0.8 / len(f1_cols) if f1_cols else 0.2
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 2), 5))
    for i, col in enumerate(f1_cols):
        ax.bar(x + i * width, df[col], width, label=col.replace("f1_classe_", "classe "))
    ax.set_xticks(x + width * (len(f1_cols) - 1) / 2)
    ax.set_xticklabels(df["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Benchmark — F1 par classe")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(_ARTIFACTS_DIR, "benchmark_f1_par_classe.png"), dpi=150)
    plt.close(fig)

    # Graphe ressources (uniquement si profils fournis)
    if profiles:
        perf_metrics = [c for c in ["train_time_s", "inference_time_per_sample_ms",
                                     "ram_inference_peak_mb"] if c in df.columns]
        fig, axes = plt.subplots(1, len(perf_metrics), figsize=(5 * len(perf_metrics), 4))
        if len(perf_metrics) == 1:
            axes = [axes]
        for ax, col in zip(axes, perf_metrics):
            vals = df[col].fillna(0).infer_objects(copy=False)
            ax.bar(range(len(df)), vals, color="steelblue")
            ax.set_xticks(range(len(df)))
            ax.set_xticklabels(df["model"], rotation=15, ha="right")
            ax.set_title(col.replace("_", " "))
            ax.set_ylim(0, vals.max() * 1.2 if vals.max() > 0 else 1)
        plt.suptitle("Benchmark — ressources")
        plt.tight_layout()
        plt.savefig(os.path.join(_ARTIFACTS_DIR, "benchmark_ressources.png"), dpi=150)
        plt.close(fig)

    return df
