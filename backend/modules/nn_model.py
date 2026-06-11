# =============================================================================
#  MODÈLE DE RÉSEAU DE NEURONES
#  Création, entraînement et inférence du modèle Keras.
# =============================================================================

import os

_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
os.makedirs(_ARTIFACTS_DIR, exist_ok=True)



def train_model(
    model,
    X, y,
    X_val=None, y_val=None,
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=None,
):
    """
    Entraîne le modèle et retourne (model, history).

    Si X_val et y_val sont fournis, la validation est activée à chaque époque.
    """
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

    history = model.fit(
        X, y,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
    )
    return model, history


# ── Prédiction ────────────────────────────────────────────────────────────────

def model_predict(model, X):
    """Retourne les prédictions aplaties en tableau 1-D."""
    return model.predict(X).flatten()


