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

## Limite observée en test : fragilité lexicale du TF-IDF + effet du seuil

**Observation** : avec des constantes vitales identiques, la description change tout — et de façon contre-intuitive :

| Description | Proba 0 | Proba 1 | Proba 2 | Prédiction |
|---|---|---|---|---|
| "Simple demande d'information **sur les horaires de garde**" | 0.98 | 0.005 | 0.015 | Pas urgent |
| "Simple demande d'information" (tronquée) | 0.52 | 0.455 | 0.025 | **Urgent** |

Raccourcir la phrase fait passer la prédiction de "Pas urgent" (sûr à 98 %) à "Urgent".

**Pourquoi (1 — fragilité lexicale du TF-IDF)** : le TF-IDF est un sac-de-mots, il ne comprend pas le sens global mais seulement des mots isolés. Le signal "non-urgent" vivait dans des mots précis : `horaires`, `garde`, `de garde`, `horaires de`. Ce sont des marqueurs administratifs dans le dataset (planning, garde médecin → classe 0). En tronquant la phrase, on retire exactement ces mots. Il ne reste que `simple`, `demande`, `information` — des mots **génériques et ambigus** (on "demande" / veut de l'"information" dans tous les contextes, urgents compris). Le modèle perd sa certitude : proba classe 0 chute de 0.98 → 0.52, classe 1 monte de 0.005 → 0.455.

**Pourquoi (2 — effet du seuil)** : en argmax, 0.52 > 0.455 → ça resterait "Pas urgent". Mais le seuil abaissé `classe 1 ≥ 0.20` est franchi par 0.455 → la décision est **forcée** à "Urgent" alors que "Pas urgent" reste majoritaire. Les deux leviers se cumulent.

**Ce que ça révèle** : vérifié concrètement en chargeant `S2_tfidf.pkl` — `horaires de garde` (trigramme) n'est même pas dans le vocabulaire, mais `horaires`, `garde`, `de garde` y sont et portent le signal. La normalisation L2 fait que la version tronquée donne des poids plus élevés (0.461 vs 0.276) à ses 5 mots restants, mais ça n'aide pas : ces mots sont peu discriminants.

**À mentionner en soutenance** : exemple parfait des deux mécanismes du système réunis — (1) la fragilité du TF-IDF qui dépend de mots-clés précis et ne gère ni le sens ni la négation, (2) l'effet du seuil anti-sous-triage qui bascule une décision pourtant majoritairement "non-urgente". Piste d'amélioration : embeddings / modèle de langage pour capturer le sens plutôt que des mots isolés (au prix de l'explicabilité).

## Recommandations — réglage des hyperparamètres

**Principe directeur** : ne pas tuner "pour tuner". Concentrer l'effort là où (1) l'impact est réel ET (2) on peut l'expliquer, en s'appuyant sur ce qu'on a observé dans le projet. À garder en tête : les hyperparamètres donnent des gains **de second ordre** — le vrai levier de contrôle du sous-triage reste le **seuil**, pas les hyperparamètres.

### Reco 1 — Augmenter `max_features` du TF-IDF (500 → 1000-2000) + `min_df=2`

**Pourquoi** : c'est la seule reco qui attaque les bugs réellement observés ("ok"/"je vais bien" → vecteur vide ; "horaires de garde" → trigramme absent du vocabulaire). Cause racine commune : vocabulaire trop petit (500 mots) → beaucoup de mots porteurs de sens sont ignorés. Élargir = moins de descriptions "invisibles". `min_df=2` filtre les mots ultra-rares (bruit) qu'on récupérerait en élargissant.
**Bémol** : plus de colonnes → entraînement plus lent + léger risque d'overfit, à surveiller en CV.
**Reco n°1 car** : impact observable et explicable (démarche : bug constaté → cause diagnostiquée → correction).

### Reco 2 — `RandomizedSearchCV` sur XGBoost (pas RandomForest)

**Pourquoi XGBoost** : (1) c'est là que le tuning paie le plus (`learning_rate`, `max_depth`, `subsample`, `colsample` interagissent et débloquent de vrais gains) ; (2) XGBoost était déjà le meilleur sur la classe 2 sans pénalisation → on construit sur une force.
**Pourquoi Randomized et pas Grid** : GridSearch teste TOUTES les combinaisons (explosion combinatoire). RandomizedSearch tire N combinaisons au hasard (~90 % du gain pour ~5 % du temps).
**Point crucial — le scorer** : optimiser un scorer = **recall classe 2** (ou coût-sensible), JAMAIS l'accuracy (trompeuse : déséquilibre + coût asymétrique). Sinon on règle finement pour la mauvaise métrique.

### Reco 3 — Sur RandomForest (modèle de prod) : `min_samples_leaf` / `max_depth`, PAS `n_estimators`

**Pourquoi pas `n_estimators`** : à 200 arbres on est dans les rendements décroissants. Monter coûte du temps d'inférence (compte en télémédecine) pour un gain quasi nul. Fausse bonne idée fréquente.
**Pourquoi `min_samples_leaf` (2-5) / `max_depth`** : la matrice TF-IDF est haute dimension et creuse → terrain à sur-apprentissage (feuilles sur un seul patient). Limiter la profondeur / forcer des feuilles à plusieurs patients = régularisation → meilleure généralisation.
**Effet attendu** : gain modeste mais réel sur la stabilité (barres d'erreur CV plus resserrées).

### Reco 4 — Ne PAS tuner le réseau de neurones

**Pourquoi** : (1) forte variance (init aléatoire → dur de distinguer gain réel du hasard) ; (2) sur données tabulaires + sparse, les arbres dominent (résultat empirique établi) ; (3) dur à justifier à l'oral. Le NN reste utile comme **point de comparaison** dans le benchmark, mais l'optimiser = temps mal investi.

## Comment détecter l'overfitting sur les modèles

**Principe** : un modèle sur-appris est excellent sur les données vues (train) mais moins bon sur des données nouvelles (test). On cherche un **écart**.

**Méthode 1 — Écart train vs test (la plus directe, PAS encore en place)**
Mesurer la même métrique sur train ET test. Train ≈ Test = bonne généralisation ; Train très haut + Test bien plus bas = overfit ; les deux bas = sous-apprentissage. Piège : un RandomForest non bridé (`max_depth=None`, `min_samples_leaf=1`) atteint ~100 % sur le train quasi systématiquement → ça seul ne prouve rien, c'est la TAILLE de l'écart qui compte. À ajouter : le code calcule les métriques uniquement sur `y_test`.

**Méthode 2 — Stabilité en cross-validation (déjà en place : `compare_cv.png`)**
Le `std` des folds = indice d'overfit. Barres d'erreur courtes = résultats cohérents (robuste) ; barres longues = modèle très sensible au découpage = tendance au sur-apprentissage.

**Méthode 3 — Courbes de loss du NN (déjà en place : MLflow)**
`train_loss` et `val_loss` par epoch. Les deux baissent ensemble = OK. `train_loss` descend mais `val_loss` stagne/remonte = overfit qui commence. Protection déjà présente : `EarlyStopping(restore_best_weights=True)` arrête dès que `val_loss` cesse de s'améliorer.

## Normalisation (MinMaxScaler) — choix et bémol

**Choix** : le préprocessing fait de la **normalisation** (`MinMaxScaler` → valeurs dans [0,1]), pas de la standardisation (`StandardScaler` → moyenne 0 / écart-type 1). Présent dans `choix_model_inline.py` ET `modules/preprocess.py`.

**Pourquoi MinMax** :
1. **Fusion avec le TF-IDF** (raison la plus forte) : les valeurs TF-IDF sont positives et bornées ~[0,1]. MinMax met les constantes vitales sur la **même échelle positive** → matrice `hstack` homogène. StandardScaler donnerait des valeurs négatives/non bornées qui jurent avec le bloc texte (et casse la sparsité en centrant).
2. **Variables à bornes physiques connues** (SpO₂ [0,100], temp clippée…) → adapté au min-max.
3. **Effet selon modèles** : arbres (RF/XGBoost/LightGBM) **insensibles** à l'échelle (décisions par seuils) → neutre pour le modèle de prod. LogReg et NN, eux, **ont besoin** d'une mise à l'échelle ; le NN apprécie des entrées bornées [0,1].

**Le bémol — MinMax est sensible aux valeurs extrêmes** :
- *Compression* : une valeur extrême étire le [0,1] et comprime les valeurs normales dans une bande plus étroite.
- *Hors plage à l'inférence* : MinMax apprend min/max sur le train. En production (API), une valeur plus extrême que tout le train est mappée **> 1** (pas de plafond).

**Est-ce un problème ? En théorie oui, en pratique quasiment pas** : le modèle déployé est un **RandomForest**, insensible à l'échelle → les deux risques sont neutralisés. Le bémol ne deviendrait réel qu'en déployant le **NN ou la LogReg**.

**Comment faire si ça posait problème** (donc surtout NN/LogReg) :
- `RobustScaler` (médiane + écart interquartile) → robuste aux outliers. Inconvénient : perd la propriété bornée [0,1], donc l'homogénéité avec le TF-IDF.
- Clipper à un percentile clinique avant MinMax → à éviter ici car on veut *garder* les extrêmes médicaux (signal, pas bruit).
- **Décision** : ne rien changer tant que le modèle de prod est un arbre.

**Phrase oral** : "On normalise en MinMax pour rester sur la même échelle positive que le TF-IDF ; c'est neutre pour les arbres et utile pour LogReg/NN. MinMax est sensible aux extrêmes, mais on l'assume car ce sont des extrêmes cliniques réels et notre modèle de prod (arbre) y est insensible."

## Ce que fait le `class_weight` (synthèse)

- **Quoi** : pendant l'entraînement, multiplie la contribution à la *loss* de chaque échantillon par le poids de sa **vraie classe**. Une erreur sur une classe lourde coûte N× plus.
- **Effet** : déplace la frontière de décision **vers les classes surpondérées** → le modèle prédit ces classes plus facilement.
- **Quand** : agit uniquement à l'**entraînement** (pas à la prédiction — ça, c'est le rôle des seuils).
- **Conséquence métier** : recall des classes lourdes ↑, précision ↓ (plus de fausses alertes) → sur-triage assumé.
- **Ne PAS confondre** avec `sample_weight` (même effet mais poids par échantillon, voir section dédiée) ni avec les **seuils** (levier de prédiction, déterministe).

## `class_weight` agit sur le recall, pas sur la précision — pourquoi (question piège jury)

**Le principe** : `class_weight` n'agit pas "sur le recall" directement, il pondère la **loss d'entraînement**. Détail clé : **le poids est porté par la VRAIE classe de l'échantillon.**

- Un **faux négatif de classe 2** (vrai 2 prédit 0/1 = cas vital raté) vient d'un échantillon de vraie classe 2 → il porte le poids **15** → fortement puni → le modèle apprend à les éviter → **recall classe 2 ↑** (recall = VP/(VP+FN)).
- Un **faux positif de classe 2** (vrai 0/1 prédit 2 = fausse alerte) vient d'un échantillon de vraie classe 0 ou 1 → il porte le poids **1 ou 6**, PAS 15 → non pénalisé → rien ne l'empêche → **précision classe 2 ↓** (precision = VP/(VP+FP)).

**Sur la matrice de confusion** : `class_weight` agit sur la **ligne 2** (faux négatifs = recall ; nos matrices sont `normalize="true"` = recall par ligne), pas sur la **colonne 2** (faux positifs = précision). D'où l'échange recall↑ / précision↓ = le sur-triage assumé.

**Corollaire (observé en test)** : augmenter les poids en bloc n'augmente pas forcément le recall classe 2 — ce qui compte est le **ratio classe 2 / classe 1**. Passer de `{1:6, 2:15}` (ratio 2,5) à `{1:10, 2:20}` (ratio 2,0) **baisse** l'avantage relatif de la classe 2 → certains vrais cas-2 limites basculent en classe 1 → recall classe 2 stagne ou baisse (le reste étant du bruit, on est proche du plafond). Pour pousser le recall classe 2 : augmenter le **ratio** (ex. `{1:6, 2:25}`), pas les poids absolus.

## Détermination des seuils — méthodologie

**Principe** : un seuil traduit une proba en décision. Le baisser = recall ↑ mais précision ↓. Pas de "meilleur" absolu → on définit un objectif, puis on trouve le point qui le satisfait au moindre coût.

**Règle d'or — ne JAMAIS régler les seuils sur le test rapporté** (fuite de données). Trois usages distincts :
1. Train → entraîne le modèle
2. Validation (ou cross-validation) → règle les seuils
3. Test → mesure finale, une fois, seuils figés
Les seuils se règlent **post-entraînement** sur les `predict_proba` → pas de ré-entraînement, on en teste des centaines en quelques secondes.

**Deux façons de formuler l'objectif** :
- **Par contrainte** (notre cas) : fixer recall classe 2 cible (98-100 %), minimiser le sur-triage sous cette contrainte.
- **Par fonction de coût** (le plus défendable) : matrice de coût (ex. rater un vital 2→0 = 100, sous-trier = 10, sur-trier = 1) → choisir les seuils qui minimisent le coût total. Encode explicitement la priorité médicale.

**Recette (approche par contrainte)** :
1. `predict_proba` sur la validation.
2. Cible : recall classe 2 (ex. 98 %).
3. `seuil_2` = percentile des `proba_2` des vrais cas-2 correspondant à la cible (100 % → min ; 98 % → 2ᵉ percentile). + marge de sécurité.
4. `seuil_1` = idem pour la classe 1 sur les cas restants.
5. Vérifier le coût (précision classe 0, sur-triage) ; relâcher la contrainte si trop cher.
6. Figer, mesurer une fois sur le test.

**Cascade 2 seuils** : on teste `seuil_2` puis `seuil_1`. Réglage séquentiel (seuil_2 d'abord = priorité vitale, puis seuil_1) cohérent avec la hiérarchie de gravité. Alternative rigoureuse : grille 2D `(seuil_2, seuil_1)`.

**Robustesse** : un seuil donnant `[2][0]=0` sur la validation peut laisser passer un vital sur de nouvelles données. Protections : cross-validation (valeur stable, pas un seul fold) + marge (seuil un peu sous le min observé).

## Les seuils dépendent du modèle ET du scénario (limite actuelle)

**Oui, les seuils optimaux changent selon le modèle et le scénario** — car chacun produit des probas distribuées différemment :
- **Modèle** : RF (probas étalées) vs XGBoost/LightGBM (plus confiants) vs NN softmax (très piqué) vs LogReg. Un seuil de 0,15 n'attrape pas la même fraction de cas-2 selon le modèle = question de **calibration**.
- **Scénario** : S1/S2 (beaucoup de signal → probas tranchées) vs S3/S4 (moins de signal → probas diffuses). Le seuil qui annule `[2][0]` en S1 ≠ celui de S4.

**Limite du code actuel** : un seul `THRESHOLDS` global appliqué à tous les modèles et scénarios. Pour la comparaison, c'est légèrement injuste (avantage le modèle dont la calibration colle par hasard au seuil) — même nature de problème que le scaler partagé.

**Approche rigoureuse** : régler les seuils **par modèle**, sur la validation, pour atteindre un **recall classe 2 cible commun** → comparer tous les modèles au même point de fonctionnement, et voir lequel a le meilleur coût (précision classe 0). Chaque modèle évalué à son meilleur.

**Décision projet** :
- **Déploiement** (un seul modèle, S2 + RandomForest) : régler les seuils spécifiquement pour lui sur la validation.
- **Benchmark** : seuil global acceptable SI assumé explicitement ("politique de décision identique"), sinon seuils par modèle à recall vital égal.

**Phrase oral** : "Les seuils optimaux dépendent du modèle (calibration) et du scénario (séparabilité). On a utilisé un seuil global pour comparer sous une politique identique, mais pour le modèle déployé on règle les seuils spécifiquement sur la validation. Comparer chaque modèle à son propre seuil à recall vital égal serait l'étape suivante."

## Questions Thomas
- Est ce qu'un KNN est pertinant ?
- Que faire de la classe intermédiaire ?
