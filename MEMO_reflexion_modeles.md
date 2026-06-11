# Mémo — Réflexion choix des modèles (scénario 1_complet.py)

## Ce qu'on a établi

- Problème de **classification multi-classes** (0 / 1 / 2), pas de régression
- Classes ordonnées ET indépendantes (degrés d'urgence distincts)

## Point clé : matrice de coût asymétrique

La direction de l'erreur compte — sous-triage toujours plus grave que sur-triage :

| Vrai \ Prédit | 0 | 1 | 2 |
|--------------|---|---|---|
| **0** | ✓ | acceptable | sur-triage |
| **1** | dangereux | ✓ | acceptable |
| **2** | catastrophique | dangereux | ✓ |

- Prédire **trop haut** → sur-triage → coûteux mais pas dangereux
- Prédire **trop bas** → sous-triage → potentiellement fatal

## Questions à reprendre

1. **Métriques** : l'accuracy traite toutes les erreurs pareil → quelle métrique différencie les classes ?
2. **Entraînement** : certains modèles acceptent une matrice de coût explicite (`class_weight`, `sample_weight`) → utile ici ?
3. **Données sparse** : après TF-IDF, matrice creuse avec milliers de colonnes → tous les algos se comportent-ils pareil ?
4. **Interprétabilité** : choix performance > explicabilité — à nuancer en contexte médical
