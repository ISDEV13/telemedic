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

## `class_weight` vs `sample_weight` — quelle différence ?

Ces deux mécanismes servent à **pénaliser différemment les erreurs selon la classe**, mais ils opèrent à des niveaux différents.

### `class_weight` — poids déclaré au niveau du modèle

On déclare une fois la pondération par classe dans le constructeur du modèle. Scikit-learn convertit automatiquement ça en poids par sample à l'intérieur de `fit()`.

```python
RandomForestClassifier(class_weight={0: 1, 1: 6, 2: 15})
# Scikit-learn fait la conversion en interne :
# chaque sample classe 2 → poids 15, classe 1 → poids 6, etc.
```

**Avantages :** simple, lisible, aucun calcul à faire soi-même.  
**Limite :** le poids est forcément le même pour tous les samples d'une même classe.

### `sample_weight` — poids déclaré au niveau de `fit()`

On passe manuellement un tableau de poids de longueur `n_samples`, un poids par observation. Ça offre un contrôle total — deux samples de la même classe pourraient avoir des poids différents.

```python
sw = np.array([class_weight[int(y)] for y in y_train])
model.fit(X_train, y_train, sample_weight=sw)
```

**Avantages :** contrôle granulaire, nécessaire pour les algos qui ne supportent pas `class_weight` nativement (XGBoost, par exemple).  
**Cas d'usage avancé :** on pourrait pondérer différemment des samples rares dans une classe (ex. cas vitaux avec symptômes atypiques).

### Pourquoi XGBoost passe par `sample_weight` dans notre code

XGBoost n'expose **pas** de paramètre `class_weight` dans son constructeur. On fait donc la conversion manuellement avant `fit()` — c'est exactement ce que scikit-learn fait en interne pour RandomForest ou LogisticRegression, rendu visible.

### Résultat : identique dans notre cas

Comme on veut un poids uniforme par classe (tous les vitaux pèsent 15, tous les urgents pèsent 6), les deux approches donnent le **même résultat mathématique**. La différence est purement d'API.

| | `class_weight` | `sample_weight` |
|---|---|---|
| Où ça se passe | constructeur du modèle | paramètre de `fit()` |
| Qui calcule les poids par sample | scikit-learn en interne | nous, manuellement |
| Granularité | par classe uniquement | par sample (potentiellement unique) |
| Supporté par XGBoost | ✗ | ✓ |
| Supporté par KNN | ✗ | ✗ → raison d'exclusion |

---

## Comportement de la régression logistique face à la pénalisation

### Observation

La régression logistique est globalement moins bonne que les modèles ensemblistes. Mais après application de `class_weight = {0:1, 1:6, 2:15}`, elle **sous-classe moins** (meilleur recall classe 2) et **surclasse davantage** que les autres modèles. Est-ce cohérent ?

**Oui. C'est même une conséquence directe de sa nature linéaire.**

### Pourquoi la LogReg surclasse plus après pénalisation

La régression logistique trace un **hyperplan de décision unique** dans l'espace des features. Sans pénalisation, cet hyperplan se cale sur la distribution naturelle des données — il prédit souvent "pas urgent" (classe 0) car c'est ce qui minimise l'erreur globale sur un dataset déséquilibré.

Quand on applique `class_weight = {2: 15}`, on multiplie le gradient de la perte par 15 pour chaque erreur sur la classe 2. La LogReg n'a qu'un seul levier disponible : **déplacer globalement son hyperplan** en direction de la classe 2. Ce déplacement est brutal :

- Il réduit le sous-triage → le recall classe 2 monte
- Mais il génère plus de faux positifs "vital" → sur-triage augmente sur les classes 0 et 1

C'est un ajustement **global et indifférencié** : impossible pour un modèle linéaire de faire autrement.

### Pourquoi les modèles ensemblistes réagissent différemment

RandomForest, XGBoost et LightGBM peuvent faire des **ajustements locaux** à chaque nœud de chaque arbre. Quand on pénalise la classe 2, ils apprennent des règles précises du type :

> *"si FC > 120 ET SpO₂ < 90 ET description contient 'douleur thoracique' → vital"*

Ils augmentent le recall classe 2 **sans déplacer toute la frontière de décision**. L'ajustement est chirurgical, pas global. C'est pourquoi leurs métriques restent plus équilibrées après pénalisation.

### Tableau comparatif

| Modèle | Type de frontière | Réaction à la pénalisation |
|--------|------------------|-----------------------------|
| **LogisticRegression** | Hyperplan unique (linéaire) | Déplacement global → surclasse massivement |
| **RandomForest** | Ensemble d'arbres (local) | Ajustements ciblés par région de l'espace |
| **XGBoost / LightGBM** | Boosting (séquentiel, local) | Correction progressive, précise |
| **NeuralNetwork** | Non-linéaire (couches denses) | Intermédiaire : flexible mais sensible à l'initialisation |

### Ce que ça nous dit pour le déploiement

La LogReg est une **baseline** dans ce benchmark, pas un candidat sérieux. Elle est utile pour :
- vérifier que les autres modèles font mieux qu'un modèle simple
- visualiser de façon extrême l'effet de la pénalisation

En contexte médical, sur-classer (sur-triage) est **acceptable** — c'est le sens de la contrainte métier. Mais sur-classer **massivement** surcharge les équipes et réduit la crédibilité du système. Un bon modèle trouve le bon compromis : recall classe 2 élevé **sans** exploser le taux de faux positifs sur les autres classes.

---

## Arguments en faveur de XGBoost ou Random Forest pour le déploiement

### En faveur de XGBoost

XGBoost performe bien sur la classe 2 **même sans pénalisation**. Il a naturellement appris à isoler les cas vitaux — la pénalisation ne fait que renforcer quelque chose qu'il faisait déjà. En production, un modèle qui n'a pas besoin d'être "forcé" pour détecter les urgences vitales est plus robuste : si les poids changent, il ne s'effondre pas.

Argument supplémentaire : **vitesse d'inférence** (~0.004 ms/patient, très peu de RAM). En télémédecine avec des pics de connexion, ça compte.

### En faveur de Random Forest

Son principal avantage est la **stabilité**. Le bagging (entraîner chaque arbre sur un sous-échantillon aléatoire) réduit la variance — RF est moins sensible à un patient atypique ou à un léger changement dans les données d'entraînement. En contexte médical où les distributions peuvent dériver avec le temps, c'est une propriété importante.

Il supporte aussi `class_weight` nativement sans conversion manuelle, et son comportement est plus facile à auditer si on doit justifier une décision devant un médecin.

### L'argument commun aux deux

Ni RF ni XGBoost ne sacrifient recall classe 1 et précision classe 0 pour gagner sur classe 2, contrairement à LogReg. Ils maintiennent un **équilibre sur les trois métriques** — diagonale TP correcte avec un minimum d'erreurs de sous-triage, pas "recall classe 2 à 1.00 au prix de tout le reste".

| Critère | XGBoost | RandomForest |
|---------|---------|--------------|
| Recall classe 2 sans péna | Meilleur naturellement | Bon |
| Équilibre des métriques | Oui | Oui |
| Vitesse d'inférence | Très rapide | Moyen |
| Stabilité / variance | Moyenne | Élevée |
| Auditabilité médicale | Moyenne | Bonne |
| `class_weight` natif | Non (→ `sample_weight`) | Oui |

**Conclusion** : XGBoost pour la performance brute et la vitesse. RandomForest pour la stabilité et l'auditabilité. Les deux sont des candidats sérieux pour un déploiement réel, contrairement à LogReg.

---

## Questions à reprendre

1. **Métriques** : l'accuracy traite toutes les erreurs pareil → quelle métrique différencie les classes ?
2. **Entraînement** : certains modèles acceptent une matrice de coût explicite (`class_weight`, `sample_weight`) → utile ici ?
3. **Données sparse** : après TF-IDF, matrice creuse avec milliers de colonnes → tous les algos se comportent-ils pareil ?
4. **Interprétabilité** : choix performance > explicabilité — à nuancer en contexte médical

## Limite observée en test : une feature seule ne suffit pas

**Observation** : température à 34°C (hypothermie) → modèle prédit "Urgent" (1) au lieu de "Très urgent" (2).

**Pourquoi** : le modèle ne raisonne pas comme un médecin qui verrait 34°C et conclurait immédiatement. Il combine **toutes les features ensemble** pour décider. Si les autres vitals sont normaux, la proba de classe 2 peut rester sous le seuil de 0.15.

**Ce que ça révèle** : le modèle a appris des corrélations statistiques depuis le dataset — dans les données d'entraînement, 34°C avec des autres vitals normaux n'était probablement pas systématiquement étiqueté "Très urgent". Ce n'est pas un bug, c'est une limite inhérente à l'approche ML par rapport à un système expert à règles.

**À mentionner en soutenance** : pour pousser la prédiction à "Très urgent", il faut combiner plusieurs signes défavorables (température basse + saturation basse + fréquence cardiaque anormale). C'est cohérent avec la réalité clinique mais peut poser problème si un patient présente un seul signe isolé très grave.

**Biais de couverture du dataset** : le profil "vital" appris par le modèle est tachycardie + fièvre + hypoxie + hypertension (moyennes classe 2 : FC ~120, temp ~39°C, SpO₂ ~88%, tension ~175). L'**hyperthermie est bien couverte**, l'**hypothermie (ex: 30°C + 20 bpm) ne l'est pas** — le modèle la classe en "Pas urgent" car ce pattern n'existe pas ou peu dans les données. En production, il faudrait un dataset généré avec des médecins pour couvrir tous les tableaux cliniques critiques.

## Questions Thomas
- Est ce qu'un KNN est pertinant ?
- Que faire de la classe intermédiaire ?
