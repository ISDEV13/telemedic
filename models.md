# Guide exhaustif des modèles de Machine Learning

> Référence pédagogique : ce que fait chaque modèle, **pourquoi ça marche**, pour quoi, et sur quelles données.
> Chaque modèle inclut un schéma et un exemple concret.

---

## Sommaire

1. [Modèles linéaires](#1-modèles-linéaires)
2. [Modèles à base d'arbres](#2-modèles-à-base-darbres)
3. [Méthodes ensemblistes](#3-méthodes-ensemblistes)
4. [Modèles à noyaux — SVM](#4-modèles-à-noyaux--svm)
5. [Modèles probabilistes — Naive Bayes](#5-modèles-probabilistes--naive-bayes)
6. [Apprentissage par similarité — KNN](#6-apprentissage-par-similarité--knn)
7. [Réseaux de neurones](#7-réseaux-de-neurones)
8. [Apprentissage non supervisé — Clustering](#8-apprentissage-non-supervisé--clustering)
9. [Réduction de dimensionnalité](#9-réduction-de-dimensionnalité)
10. [Détection d'anomalies](#10-détection-danomalies)
11. [Apprentissage par renforcement](#11-apprentissage-par-renforcement)
12. [Tableau récapitulatif](#12-tableau-récapitulatif)
13. [Données mixtes — numérique + texte](#13-données-mixtes--numérique--texte)

---

## 1. Modèles linéaires

### Régression Linéaire

**Ce qu'il fait**
Trouve la droite (ou l'hyperplan en plusieurs dimensions) qui minimise la somme des carrés des erreurs entre les prédictions et les vraies valeurs.

**Pourquoi ça marche**
Le modèle suppose que la relation entre les features et la cible est une **combinaison linéaire** des inputs. Il cherche les coefficients (poids) qui minimisent l'erreur quadratique moyenne via une formule mathématique directe (pas d'itérations nécessaires). Ça marche quand cette hypothèse linéaire est vraie ou approximativement vraie dans la plage de données observée.

```
               prix
                |          * prédiction
            400k|         /
                |        / ← droite apprise
            300k|       /
                |      /      * vrai point
            200k|     /    * erreur = écart vertical
                |    /
            100k|   /
                |  /
                +--+--+--+--+--→  surface (m²)
                  30 50 70 90

  Prix = 3500 × surface + 5000
         ↑                  ↑
      coefficient        constante (biais)
```

**Exemple concret**
```
Données :
  surface=50m²  → prix=180 000€
  surface=70m²  → prix=250 000€
  surface=90m²  → prix=320 000€

Modèle appris : Prix = 3 500 × surface + 5 000
Prédiction pour 60m² : 3500×60 + 5000 = 215 000€
```

**Pertinent pour** : prédire une valeur continue avec une relation linéaire (prix, durée, température)
**Pas pertinent si** : la relation est courbe, ou si tu veux prédire une classe

---

### Régression Logistique

**Ce qu'il fait**
Classe des exemples en calculant une probabilité d'appartenance à chaque classe. Malgré son nom, c'est bien un modèle de **classification**.

**Pourquoi ça marche**
Le modèle prend la combinaison linéaire des features (comme la régression linéaire) et la passe dans une **fonction sigmoïde** qui écrase n'importe quel nombre entre 0 et 1. Ce 0→1 représente une probabilité. L'entraînement optimise les coefficients pour que les probabilités soient les plus proches possible des vraies classes (via la log-vraisemblance).

```
  Combinaison linéaire :
  z = w1×age + w2×tension + w3×temp + biais

                     1
  Sigmoïde : σ(z) = ────────
                   1 + e^(-z)

  σ(z)
  1.0 |                    ___________
      |                  /
  0.5 |- - - - - - - - -/- - - - - - -  ← seuil de décision
      |               /
  0.0 |______________/
      +─────────────────────────────→ z
         ↑               ↑
      valeur         valeur
      très           très
      négative       positive
      = classe 0     = classe 1

  Résultat : P(urgence=1) = 0.82  → on prédit classe 1
```

**Exemple concret**
```
Patient : age=72, tension=185, temp=38.5
  z = 0.04×72 + 0.03×185 + 0.5×38.5 - 30 = 3.13
  σ(3.13) = 0.96 → 96% de probabilité d'urgence → prédit urgence
```

**Pourquoi c'est excellent avec du texte TF-IDF** : TF-IDF produit des vecteurs creux (sparse) avec des milliers de colonnes. La régression logistique apprend un poids pour chaque mot. Le mot "douleur" aura un poids positif vers "urgence", "bénin" aura un poids négatif. C'est lisible et interprétable.

**Pertinent pour** : classification tabulaire, texte vectorisé, quand l'interprétabilité compte
**Pas pertinent si** : relations non-linéaires complexes entre les features

---

### Ridge et Lasso

**Ce qu'il fait**
Régression linéaire avec une **pénalité** qui empêche les coefficients de devenir trop grands.

**Pourquoi ça marche**
Sans régularisation, si tu as 500 features pour 200 samples, le modèle peut mémoriser parfaitement les données d'entraînement (sur-apprentissage). La pénalité force le modèle à rester "humble" :
- **Ridge (L2)** : ajoute la somme des carrés des coefficients à l'erreur. Tous les coefficients restent petits mais aucun n'est annulé.
- **Lasso (L1)** : ajoute la somme des valeurs absolues. Pousse certains coefficients exactement à zéro → **sélection automatique de features**.

```
  Objectif Ridge : Minimiser  Erreur + λ × Σ(wi²)
  Objectif Lasso : Minimiser  Erreur + λ × Σ|wi|
                                         ↑
                               λ contrôle la force de pénalité

  Lasso avec λ fort :
  w_age        =  0.42   (feature importante, gardée)
  w_patient_id =  0.00   ← annulé → feature non pertinente éliminée
  w_tension    =  0.38   (gardée)
  w_zone_vie   =  0.00   ← annulé
```

**Pertinent pour** : beaucoup de features par rapport aux samples, sélection de variables (Lasso)

---

## 2. Modèles à base d'arbres

### Arbre de Décision

**Ce qu'il fait**
Pose des questions binaires sur les features de façon hiérarchique jusqu'à atteindre une décision.

**Pourquoi ça marche**
L'algorithme cherche à chaque nœud la question (feature + seuil) qui **divise le mieux les données** selon un critère de pureté (Gini ou entropie). Une division "pure" = chaque sous-groupe contient majoritairement une seule classe. Il répète ce processus récursivement jusqu'à ce que les feuilles soient pures ou qu'un critère d'arrêt soit atteint.

```
                    [sat_oxygène < 92%?]
                    /                \
                 OUI                 NON
                  |                   |
         [freq_cardiaque > 120?]   [temp > 39.5°?]
          /             \            /          \
        OUI             NON        OUI          NON
         |               |          |            |
    Urgence 2       Urgence 1   Urgence 1   Non urgent
    (vitale)        (relative)  (relative)    (0)

  Critère Gini : mesure l'impureté d'un nœud
  Gini = 0   → nœud pur (une seule classe)
  Gini = 0.5 → nœud impur (50/50 entre 2 classes)
```

**Exemple concret**
```
Patient : sat_oxygène=88%, freq_cardiaque=130
  → sat < 92% ? OUI → aller à gauche
  → freq > 120 ? OUI → aller à gauche
  → Prédiction : Urgence vitale (2)
```

**Pertinent pour** : données tabulaires mixtes, interprétabilité maximale, règles métier lisibles
**Pas pertinent si** : données complexes → l'arbre devient trop profond et sur-apprend

---

### Random Forest

**Ce qu'il fait**
Entraîne N arbres de décision en parallèle, chacun sur un sous-échantillon différent des données et des features. Vote majoritaire final.

**Pourquoi ça marche — l'intuition clé**
Un seul arbre fait des erreurs "spécifiques" à l'échantillon sur lequel il a appris (variance élevée). Si tu prends 100 arbres entraînés sur des données légèrement différentes, leurs erreurs sont **décorrélées** et s'annulent au vote. C'est le principe de la sagesse des foules : chaque expert se trompe, mais rarement tous sur le même patient.

```
  Dataset original (2000 patients)
         |
    ┌────┴────┐
    │ Bootstrap│  ← tirage AVEC remise (certains patients apparaissent
    └────┬────┘    plusieurs fois, d'autres pas du tout)
         |
  ┌──────┼──────┐
  │      │      │   ...  100 arbres
  ↓      ↓      ↓
Arbre1 Arbre2 Arbre3   (chacun voit seulement √n features aléatoires à chaque nœud)
  ↓      ↓      ↓
  2      1      2
  └──────┼──────┘
         ↓
    Vote : 2 gagne  → Prédiction finale : Urgence 2

  Feature Importance : les features utilisées haut dans les arbres
  (qui divisent le mieux) ont une importance élevée
```

**Pourquoi la sélection aléatoire de features est cruciale** : si une feature est très dominante (ex: `sat_oxygène`), tous les arbres commenceraient par elle et seraient corrélés. Forcer chaque arbre à choisir parmi un sous-ensemble aléatoire de features casse cette corrélation.

**Pertinent pour** : tabulaire mixte, outliers, valeurs manquantes, feature importance
**Pas pertinent si** : haute dimension sparse (TF-IDF avec 50k features), inférence ultra-rapide

---

### XGBoost / LightGBM / CatBoost

**Ce qu'il fait**
Construit des arbres **séquentiellement** : chaque arbre corrige les erreurs du précédent.

**Pourquoi ça marche — l'intuition du boosting**
Imagine un étudiant qui passe un QCM. Après chaque correction, il ne révise QUE les questions qu'il a ratées. Le prochain arbre se concentre exactement sur les patients mal classés par l'ensemble des arbres précédents. Mathématiquement, chaque arbre apprend le **gradient de l'erreur résiduelle** (d'où "Gradient Boosting").

```
  Itération 1 : Arbre 1 prédit grossièrement
  ┌─────────────────────────────────────────┐
  │ Vrai : [2, 0, 1, 2, 0]                 │
  │ Prédit: [1, 0, 1, 1, 0]                │
  │ Résidu: [+1, 0, 0, +1, 0]  ← erreurs   │
  └─────────────────────────────────────────┘
         ↓
  Itération 2 : Arbre 2 apprend à corriger les résidus
  ┌─────────────────────────────────────────┐
  │ Arbre 2 se concentre sur les patients   │
  │ avec résidu ≠ 0 (patients 1 et 4)       │
  └─────────────────────────────────────────┘
         ↓
  Prédiction finale = Arbre1 + 0.1×Arbre2 + 0.1×Arbre3 + ...
                              ↑
                        learning rate : chaque arbre contribue peu
                        pour éviter le sur-apprentissage

  Différences XGBoost / LightGBM :
  XGBoost    : divise les arbres par NIVEAU (breadth-first)
  LightGBM   : divise par FEUILLE (leaf-wise, plus rapide, souvent meilleur)
  CatBoost   : encode les catégorielles AVANT l'entraînement (ordered boosting)
```

**Pourquoi c'est le meilleur sur tabulaire** : le gradient boosting peut approximer n'importe quelle fonction complexe. Sur des données structurées avec des interactions entre features (ex: tension élevée ET saturation basse = urgence vitale), il capture ces interactions implicitement via les arbres successifs.

**Pertinent pour** : tabulaire structuré, meilleur modèle pour la plupart des compétitions Kaggle sur données tabulaires
**Pas pertinent si** : images, audio, texte brut (séquences)

---

## 3. Méthodes ensemblistes

### Bagging

**Ce qu'il fait**
Entraîne plusieurs instances du **même modèle** sur des sous-échantillons différents (bootstrap), puis moyenne les prédictions.

**Pourquoi ça marche**
Réduit la **variance** : un modèle instable (qui change beaucoup selon les données) devient stable quand on moyenne ses prédictions sur plusieurs tirages. Random Forest est du bagging appliqué aux arbres.

```
  Données originales : 1000 samples
        ↓  Bootstrap (tirage avec remise)
  ┌─────┐  ┌─────┐  ┌─────┐
  │800s │  │800s │  │800s │  (certains répétés, certains absents)
  └──┬──┘  └──┬──┘  └──┬──┘
     │        │        │
  Modèle1  Modèle2  Modèle3
     │        │        │
     └────────┼────────┘
              ↓
         Moyenne / Vote
```

---

### Boosting

**Ce qu'il fait**
Entraîne des modèles **en séquence**, chacun se concentrant sur les erreurs du précédent.

**Pourquoi ça marche**
Réduit le **biais** : un modèle simple (arbre peu profond = "weak learner") devient complexe quand on cumule des dizaines de corrections successives. Voir XGBoost ci-dessus pour le détail.

---

### Stacking

**Ce qu'il fait**
Entraîne des modèles de niveau 1 (Random Forest, XGBoost, SVM...), puis un **méta-modèle** apprend à combiner leurs prédictions.

**Pourquoi ça marche**
Chaque modèle a des forces différentes. Random Forest peut être excellent sur les features numériques, la régression logistique sur le texte. Le méta-modèle apprend **quand faire confiance à qui**.

```
  Input patient
      ↓
  ┌───────────────────────────┐
  │  Niveau 1 (base models)   │
  │  ┌──────┐ ┌─────┐ ┌───┐  │
  │  │  RF  │ │ XGB │ │ LR│  │
  │  └──┬───┘ └──┬──┘ └─┬─┘  │
  └─────┼────────┼───────┼───┘
        ↓        ↓       ↓
      [0.8]    [0.7]   [0.9]   ← prédictions de niveau 1
        └────────┼───────┘
                 ↓
         [Méta-modèle LR]     ← apprend à combiner
                 ↓
          Prédiction finale
```

**Pertinent pour** : compétitions ML, extraction du dernier % de performance
**Pas pertinent si** : production (trop complexe à maintenir et expliquer)

---

## 4. Modèles à noyaux — SVM

### SVM — Support Vector Machine

**Ce qu'il fait**
Trouve l'hyperplan qui sépare les classes avec la **marge maximale**. Seuls les points les plus proches de la frontière (les "vecteurs de support") définissent cet hyperplan.

**Pourquoi ça marche — l'intuition de la marge**
Une grande marge = plus de tolérance aux données bruitées. Si la frontière passe trop près de certains points d'entraînement, le moindre bruit déplacera un point de l'autre côté. Maximiser la marge maximise la robustesse.

**Pourquoi les noyaux changent tout**
Parfois les classes ne sont pas séparables par une droite. Le **noyau** (kernel trick) projette implicitement les données dans un espace de plus haute dimension où elles deviennent séparables, **sans calculer explicitement cette projection** (très économique en mémoire).

```
  Données non séparables en 2D :      Après noyau RBF (projection en 3D) :
                                          ↑ z
  ●●●●●●                                  |      ○○○○○
  ●●○○●●    ← mélangées                   |   ●●●●●●●●
  ●●○○●●                              ────┼────────────→ x
  ●●●●●●                                  |
                                          |

  Support Vectors = les points les plus proches de la frontière
  ↕ marge ↕
  ─────────────────────
          ×  ×          ← hyperplan optimal
  ─────────────────────

  LinearSVC + TF-IDF :
  Chaque mot = une dimension
  "douleur thoracique" → vecteur sparse de 50 000 dimensions
  SVM trouve la frontière qui sépare "urgence" / "non urgence"
  dans cet espace de mots
```

**Pertinent pour** : texte TF-IDF (LinearSVC = très efficace), petits datasets bien séparés
**Pas pertinent si** : très grand dataset, besoin de probabilités calibrées

---

## 5. Modèles probabilistes — Naive Bayes

### Naive Bayes

**Ce qu'il fait**
Calcule la probabilité de chaque classe étant donné les features observées, en utilisant le théorème de Bayes.

**Pourquoi ça marche — le théorème de Bayes**
```
  P(classe | features) = P(features | classe) × P(classe)
                         ─────────────────────────────────
                                  P(features)

  Traduit :
  P(urgence vitale | "douleur thoracique intense") =
      P("douleur thoracique intense" | urgence vitale) × P(urgence vitale)
      ────────────────────────────────────────────────────────────────────
                              P("douleur thoracique intense")
```

**L'hypothèse "naïve"** : on suppose que chaque feature est **indépendante des autres** étant donné la classe. C'est rarement vrai (tension et fréquence cardiaque sont corrélées), mais ça simplifie énormément le calcul et fonctionne étonnamment bien en pratique.

```
  Exemple avec texte :
  Texte : "patient inconscient douleur forte"

  P(urgence=2 | texte) ∝  P("inconscient"|urgence=2) ×
                           P("douleur"|urgence=2)     ×
                           P("forte"|urgence=2)       ×
                           P(urgence=2)

  P(urgence=0 | texte) ∝  P("inconscient"|urgence=0) ×  ← très faible
                           P("douleur"|urgence=0)     ×
                           P("forte"|urgence=0)       ×
                           P(urgence=0)

  → urgence=2 gagne car "inconscient" n'apparaît presque jamais
    dans les cas non urgents
```

**Pertinent pour** : classification de texte rapide, spam, catégorisation de documents, baseline NLP
**Pas pertinent si** : features fortement corrélées, relations complexes

---

## 6. Apprentissage par similarité — KNN

### K-Nearest Neighbors

**Ce qu'il fait**
Mémorise tout le dataset. Pour prédire, trouve les K points d'entraînement les plus proches et vote.

**Pourquoi ça marche**
Hypothèse : des points proches dans l'espace des features ont tendance à avoir la même classe. Si les 5 patients les plus similaires à un nouveau patient sont tous urgents, ce nouveau patient est probablement urgent aussi.

```
  Espace 2D (tension × sat_oxygène) :

  Tension
    |
185 |     ★          ← nouveau patient
    |  ②  ①  ②      ← ses 5 voisins : 2 urgences relatives (①),
    |    ①           2 urgences vitales (②)
    |                K=5 → vote : 2 fois ② vs 2 fois ① vs ... → égalité → dépend de K
165 |
    +──────────────→ Sat O2
      88  92  96

  Distance euclidienne : d = √((t1-t2)² + (s1-s2)²)
```

**Pourquoi la haute dimension le tue — la malédiction de la dimensionnalité**
En 2D, "proche" a un sens intuitif. En 10 000 dimensions (TF-IDF), tous les points sont à peu près à la même distance les uns des autres. La notion de "voisin" s'effondre.

```
  En 2D :   distance min=1,  distance max=10   → ratio 1:10
  En 100D : distance min=98, distance max=100  → ratio 1:1.02
            → tout le monde est "loin" de la même façon
```

**Pertinent pour** : petits datasets, faible dimensionnalité, systèmes de recommandation
**Pas pertinent si** : haute dimension, grand dataset, coût asymétrique, inférence rapide

---

## 7. Réseaux de neurones

### MLP — Perceptron Multicouche

**Ce qu'il fait**
Empile des couches de neurones entièrement connectées. Chaque neurone applique une transformation non-linéaire.

**Pourquoi ça marche — l'approximation universelle**
Le théorème d'approximation universelle dit qu'un MLP avec une seule couche cachée suffisamment large peut approximer **n'importe quelle fonction continue**. En pratique, plusieurs couches moins larges apprennent mieux des représentations hiérarchiques.

```
  Couche d'entrée    Couche cachée 1    Couche cachée 2    Sortie
  (8 features)       (64 neurones)      (32 neurones)      (3 classes)

  age ────────┐
  tension ────┼──→  [neurone]──→
  sat_O2 ─────┼──→  [neurone]──→  [neurone]──→  [P(0)=0.05]
  temp ───────┼──→  [neurone]──→  [neurone]──→  [P(1)=0.15]
  ...─────────┘     ...           ...        →  [P(2)=0.80]
                                                      ↑
                                               classe prédite

  Chaque neurone : z = Σ(wi × xi) + biais  →  activation = ReLU(z) = max(0, z)

  ReLU : │         /       ← introduction de la non-linéarité
         │        /          sans elle, empiler des couches
     0───┼───────/           n'aurait aucun effet (toujours linéaire)
         │      /
         └──────
```

**Pourquoi 2000 samples c'est peu** : un MLP avec 2 couches cachées peut avoir des dizaines de milliers de paramètres. Avec 2000 exemples, il y a trop peu de "contraintes" → le modèle mémorise les données au lieu d'apprendre des règles générales (sur-apprentissage).

**Pertinent pour** : tabulaire complexe avec beaucoup de données, données mixtes après concaténation
**Pas pertinent si** : petit dataset, interprétabilité requise

---

### CNN — Réseau Convolutif

**Ce qu'il fait**
Applique des filtres glissants qui détectent des patterns locaux dans des données structurées (images, séquences).

**Pourquoi ça marche — le partage de poids**
La même caractéristique (un bord, un mot-clé) peut apparaître **n'importe où** dans l'image ou le texte. Un filtre qui apprend à détecter "douleur thoracique" fonctionne qu'il soit au début ou à la fin de la phrase. En partageant le même filtre sur toute la séquence, on réduit massivement le nombre de paramètres et on gagne en généralisation.

```
  Sur une image (détection de bords) :
  ┌─────────────────────────────────┐
  │  Image 6×6 pixels               │
  │  ┌───┬───┬───┬───┬───┬───┐      │
  │  │255│255│255│  0│  0│  0│      │
  │  │255│255│255│  0│  0│  0│      │
  │  └───┴───┴───┴───┴───┴───┘      │
  │         ↓ filtre 3×3             │
  │  [-1 0 +1]                       │
  │  [-1 0 +1]  ← détecte les bords │
  │  [-1 0 +1]   verticaux           │
  └─────────────────────────────────┘

  Sur du texte (détection de n-grammes) :
  "le patient souffre de douleurs thoraciques intenses"
   ↑ filtre de taille 3 glisse sur toute la phrase
   détecte : "douleurs thoraciques intenses" → activation forte

  Pooling : garde seulement la valeur maximale
  → "peu importe où le mot-clé apparaît, on le détecte"
```

**Pertinent pour** : images (classification, détection), texte (n-grammes)
**Pas pertinent si** : données tabulaires, petit dataset

---

### RNN / LSTM / GRU

**Ce qu'il fait**
Traite les séquences élément par élément en maintenant un état caché qui résume ce qui a été vu.

**Pourquoi ça marche — et pourquoi LSTM est meilleur que RNN simple**
Un RNN simple a un problème : après de nombreuses étapes, le gradient (signal d'apprentissage) devient soit trop petit (vanishing) soit trop grand (exploding). Le LSTM ajoute des **portes** (gates) qui contrôlent explicitement quoi mémoriser et quoi oublier.

```
  RNN simple :
  "le ... patient ... souffre ... de ... douleurs ... thoraciques"
    h1 → h2 → h3 → h4 → h5 → h6
    ↑                          ↑
  début                   fin de phrase
  → h1 influence très peu h6 (gradient écrasé)

  LSTM (mémoire longue) :
                ┌────────────────────────────────┐
                │      Cellule mémoire (c)        │
                └────────────────────────────────┘
                   ↑             ↑             ↑
             [Porte oubli]  [Porte entrée]  [Porte sortie]
             "oublier ce     "écrire dans    "lire la
              qui est périmé" la mémoire"     mémoire"

  → La mémoire de "thoraciques" est préservée jusqu'à la fin
    même si la phrase est longue
```

**Pertinent pour** : séries temporelles (signaux ECG, constantes vitales dans le temps), texte séquentiel, traduction
**Pas pertinent si** : données tabulaires sans ordre, aujourd'hui remplacé par les Transformers en NLP

---

### Transformers / BERT / GPT

**Ce qu'il fait**
Traite toute la séquence en parallèle grâce à un mécanisme d'**attention** : chaque token "regarde" tous les autres tokens pour comprendre son contexte.

**Pourquoi ça marche — l'attention**
Le mot "grave" n'a pas le même sens dans "ce n'est pas grave" vs "état grave du patient". L'attention permet à chaque mot de pondérer l'importance de tous les autres mots pour construire sa représentation contextuelle.

```
  Phrase : "le patient a une douleur thoracique grave"

  Mécanisme d'attention pour le mot "grave" :
  ┌────────────┬──────────────────────────────────────┐
  │    Mot     │ Attention reçue par "grave"           │
  ├────────────┼──────────────────────────────────────┤
  │ le         │ ░░░░░░░░░░░░░░░░░░░░  (0.02)          │
  │ patient    │ ██░░░░░░░░░░░░░░░░░░  (0.10)          │
  │ a          │ ░░░░░░░░░░░░░░░░░░░░  (0.01)          │
  │ douleur    │ ████████████░░░░░░░░  (0.45) ← fort   │
  │ thoracique │ ██████░░░░░░░░░░░░░░  (0.30) ← fort   │
  │ grave      │ ██░░░░░░░░░░░░░░░░░░  (0.12)          │
  └────────────┴──────────────────────────────────────┘
  → "grave" est fortement contextualisé par "douleur thoracique"

  BERT  = encodeur  → comprendre le texte (classification)
  GPT   = décodeur  → générer du texte
  SBERT = embeddings de phrases pour mesurer la similarité
```

**Pertinent pour** : NLP avancé avec données texte riches, fine-tuning sur textes médicaux
**Pas pertinent si** : petit dataset (< 1000 samples) sans fine-tuning, ressources limitées (modèles lourds)

---

### Autoencoder

**Ce qu'il fait**
Apprend à **compresser** les données en une représentation compacte, puis à les **reconstruire**. L'objectif est que la reconstruction soit aussi fidèle que possible.

**Pourquoi ça marche pour la détection d'anomalies**
Le modèle est entraîné uniquement sur des données normales. Il apprend à bien reconstruire les cas normaux. Quand un cas anormal arrive, le modèle le reconstruit mal → erreur de reconstruction élevée → anomalie détectée.

```
  Encodeur          Espace latent       Décodeur
  (compression)     (représentation)    (reconstruction)

  [age=72      ]          ┌───┐         [age≈72      ]
  [tension=185 ] →→→→→→→→│ z │→→→→→→→→ [tension≈183 ]
  [sat_O2=88   ]    dim   │ 3 │   dim   [sat_O2≈89   ]
  [temp=38.5   ]   réduite│   │ remontée[temp≈38.4   ]
  [freq=125    ]          └───┘         [freq≈126    ]
       ↓                                      ↓
  Input (5 dim)       (2 dim)           Reconstruction
                                        Erreur = |Input - Reconstruction|

  Patient normal  : erreur = 0.02  → pas d'anomalie
  Patient anormal : erreur = 2.47  → anomalie détectée
```

**Pertinent pour** : détection d'anomalies sans labels, compression, pré-entraînement
**Pas pertinent si** : tâche de classification standard avec labels disponibles

---

### GAN — Generative Adversarial Network

**Ce qu'il fait**
Deux réseaux s'affrontent : le **Générateur** crée des données fausses, le **Discriminateur** essaie de les distinguer des vraies données.

**Pourquoi ça marche — l'analogie faussaire / détective**
Le générateur est un faussaire qui crée de fausses œuvres d'art. Le discriminateur est un expert qui détecte les faux. En s'affrontant, le faussaire s'améliore jusqu'à produire des œuvres indiscernables des vraies.

```
  Bruit aléatoire z
        ↓
  ┌───────────┐
  │ Générateur│ → Données synthétiques
  └─────┬─────┘
        │
        ↓
  ┌────────────────┐   Vraies données
  │Discriminateur  │ ←─────────────────
  └────────┬───────┘
           ↓
    "Vrai" ou "Faux" ?

  Entraînement simultané :
  - Discriminateur : maximiser sa précision à distinguer vrai/faux
  - Générateur : minimiser la capacité du Discriminateur à le détecter
```

**Pertinent pour** : génération d'images synthétiques, augmentation de données médicales rares
**Pas pertinent si** : classification standard, très difficile à entraîner (instabilité)

---

## 8. Apprentissage non supervisé — Clustering

> Ces modèles n'ont **pas de variable cible**. Ils trouvent des structures cachées.

### K-Means

**Ce qu'il fait**
Divise les données en K groupes en minimisant la distance entre chaque point et le centre de son groupe.

**Pourquoi ça marche — l'algorithme EM**
K-Means alterne entre deux étapes jusqu'à convergence :
1. **Assigner** : chaque point va au centroïde le plus proche
2. **Recalculer** : chaque centroïde se déplace vers la moyenne de son groupe

C'est garanti de converger (l'erreur totale ne peut que diminuer à chaque itération), mais peut converger vers un minimum local.

```
  Initialisation aléatoire :
  ★ = centroïde
  ● = point de données

  Étape 0            Étape 1            Convergence
  ★  ●  ●            ★→ ●  ●               ●★●
  ●  ●  ●     →      ●  ●  ●    →          ●  ●
  ●  ●  ★            ●  ●  ←★              ●  ●★
  (placement         (chaque point        (centroïdes
  aléatoire)         rejoint le plus      stables)
                     proche centroïde)

  Problème : sensible à l'initialisation
  Solution : K-Means++ choisit les centroïdes initiaux
             de façon à les espacer au maximum
```

**Pertinent pour** : segmentation client, groupes de patients similaires, exploration de données
**Pas pertinent si** : clusters non sphériques, K inconnu (utiliser Elbow Method ou Silhouette Score)

---

### DBSCAN

**Ce qu'il fait**
Trouve des clusters de **densité arbitraire** sans fixer K. Les points isolés sont automatiquement détectés comme bruit.

**Pourquoi ça marche**
Un cluster = une région dense. Un point appartient à un cluster si au moins `min_samples` voisins se trouvent dans un rayon `epsilon`. Les points sans assez de voisins proches = **anomalies**.

```
  Paramètres : ε=0.5 (rayon), min_samples=3

  ● ● ●         ← cluster 1 (dense)
  ●   ● ●
      
        ★ ★     ← cluster 2 (dense)
        ★ ★
        
  ×             ← bruit (point isolé, pas assez de voisins)

  Avantage sur K-Means :
  K-Means      DBSCAN
  ○○○ ○○○   →  ●●● ★★★   clusters détectés correctement
   ○   ○         ×         points isolés = anomalies
  (essaie de tout  (sait dire "ce point n'appartient à aucun cluster")
  assigner à un cluster)
```

**Pertinent pour** : clusters de forme irrégulière, détection d'anomalies combinée, données géographiques
**Pas pertinent si** : densité très variable entre les clusters

---

### Clustering Hiérarchique

**Ce qu'il fait**
Construit un arbre de fusion (dendrogramme) en fusionnant progressivement les points les plus proches.

**Pourquoi ça marche**
Part de N clusters (un par point) et fusionne les deux plus proches à chaque étape. Le résultat est un arbre complet qui montre les relations de similarité à toutes les granularités.

```
  4 patients : A, B, C, D

  Distances : A-B=2, C-D=3, (A,B)-(C,D)=7

  Dendrogramme :
        ┌──────────────────┐
  dist 7│              ┌───┴───┐
        │           ┌──┴──┐ ┌──┴──┐
  dist 3│           │     │ C     D
        │        ┌──┴──┐
  dist 2│        A     B

  Couper à dist=5 → 2 clusters : {A,B} et {C,D}
  Couper à dist=1 → 4 clusters : {A}, {B}, {C}, {D}
```

**Pertinent pour** : visualisation de la structure des données, exploration sans K fixé à l'avance
**Pas pertinent si** : grand dataset (O(n²) en mémoire et temps)

---

## 9. Réduction de dimensionnalité

### PCA — Analyse en Composantes Principales

**Ce qu'il fait**
Trouve les directions de **variance maximale** dans les données et projette sur un espace réduit.

**Pourquoi ça marche**
Si 8 features sont mesurées mais que 3 "axes d'information" suffisent pour expliquer 95% de la variance, PCA les trouve. Elle décompose la matrice de corrélation pour trouver les vecteurs propres (composantes principales).

```
  Données originales (2D) :
  tension
    |  . .
    | . . .   ← nuage allongé en diagonale
    |. . . .
    |  . .
    +──────── sat_O2

  Composante principale 1 (CP1) :
    |  ↗
    | ↗        ← direction de variance maximale
    |↗
    +────────

  Après PCA :
  CP1 (90% variance) : "état critique global" (tension + sat_O2 combinés)
  CP2 (9% variance)  : "différence tension/sat" (information résiduelle)

  → on peut garder CP1 seule et perdre seulement 10% d'information
```

**Pertinent pour** : réduire la dimensionnalité avant un modèle, visualisation, débruitage
**Pas pertinent si** : relations non-linéaires, données catégorielles

---

### t-SNE et UMAP

**Ce qu'il fait**
Réduit à 2-3 dimensions en préservant les voisinages locaux. Transformation non-linéaire.

**Pourquoi ça marche — t-SNE**
Modélise les distances comme des probabilités (points proches = haute probabilité d'être voisins), puis minimise la divergence entre les probabilités dans l'espace original et dans l'espace réduit.

```
  Espace original (100 dimensions)    Projection 2D (t-SNE)
  
  Points médicaux :                    ┌─────────────────┐
  urgence 0 : [features...]            │  ●●●     ★★     │
  urgence 1 : [features...]            │  ●●●    ★★★★    │  
  urgence 2 : [features...]            │         ★★      │
                                       │   ▲▲▲▲          │
                                       │    ▲▲▲           │
                                       └─────────────────┘
                                       ● classe 0  ★ classe 1  ▲ classe 2

  → si les classes forment des clusters visibles, le modèle aura
    de bonnes chances de bien les séparer
```

**Différence t-SNE vs UMAP** :
- t-SNE : meilleure séparation locale, non-déterministe, lent
- UMAP : préserve aussi la structure globale, plus rapide, utilisable en preprocessing

**Pertinent pour** : visualisation uniquement (t-SNE), preprocessing possible (UMAP)

---

## 10. Détection d'anomalies

### Isolation Forest

**Ce qu'il fait**
Construit des arbres aléatoires et mesure combien de coupures sont nécessaires pour isoler un point.

**Pourquoi ça marche**
Un point anormal est **rare et différent** : il sera vite isolé en quelques coupures aléatoires. Un point normal est entouré de voisins similaires et nécessite beaucoup de coupures pour être isolé.

```
  Dataset de constantes vitales :

  Arbre aléatoire :
  tension > 100 ?
  ├── OUI → sat_O2 > 90 ?
  │         ├── OUI → (normal, encore 8 coupures pour isoler)
  │         └── NON → ISOLÉ en 2 coupures ← anomalie potentielle
  └── NON → (normal, encore 6 coupures)

  Score d'anomalie = 2 / profondeur_moyenne_d'isolation
  Score élevé → anomalie
  Score faible → point normal

  patient_normal  : profondeur moyenne = 12.3 → score = 0.3
  patient_critique: profondeur moyenne = 2.1  → score = 0.9  ← anomalie
```

**Pertinent pour** : fraude, pannes, anomalies médicales sans labels, données tabulaires
**Pas pertinent si** : anomalies définies subjectivement ou contextellement

---

### Local Outlier Factor (LOF)

**Ce qu'il fait**
Compare la densité locale d'un point avec celle de ses voisins.

**Pourquoi ça marche**
Un point dans un endroit peu dense entouré de zones denses est local anormal, même s'il n'est pas extrême globalement.

```
  Deux clusters de patients :
  Cluster A (patients normaux)  Cluster B (patients critiques)
  ● ● ● ●                       ★ ★ ★
  ● ● ● ●                       ★ ★ ★
      
               ×    ← un seul patient isolé entre les deux groupes
                      Pas extrême globalement, mais LOF élevé
                      car ses voisins sont beaucoup plus denses
```

**Pertinent pour** : anomalies contextuelles, données avec plusieurs régions de densité différente

---

## 11. Apprentissage par renforcement

**Ce qu'il fait**
Un **agent** apprend à maximiser une récompense cumulée en interagissant avec un environnement. Il n'y a pas de dataset : l'agent génère ses propres expériences.

**Pourquoi ça marche — la boucle action-récompense**
```
  Environnement
       ↑
  Récompense (r)
  État (s)            Agent
       ↑               │
       └───────────────┘
             Action (a)

  Exemple tri d'urgence en RL :
  État     : constantes vitales actuelles du patient
  Action   : classifier niveau 0, 1 ou 2
  Récompense:
    +10 si urgence vitale correctement identifiée
    -100 si urgence vitale manquée (erreur critique)
    +5  si urgence relative correcte
    -1  si fausse alarme (0 classé en 1)

  L'agent apprend une POLITIQUE : état → action optimale
```

**Pertinent pour** : jeux, robotique, systèmes de décision séquentielle dynamique, optimisation de process
**Pas pertinent si** : dataset statique étiqueté (utiliser supervisé), environnement non simulable

---

## 12. Tableau récapitulatif

| Modèle | Supervisé | Tâche | Données idéales | Forces | Limites |
|--------|-----------|-------|-----------------|--------|---------|
| Régression Linéaire | Oui | Régression | Tabulaire, relation linéaire | Simple, interprétable | Linéaire seulement |
| Régression Logistique | Oui | Classification | Tabulaire, texte TF-IDF | Interprétable, gère sparse | Linéaire seulement |
| Ridge / Lasso | Oui | Régression | Beaucoup de features | Évite sur-apprentissage | Linéaire |
| Arbre de Décision | Oui | Classif / Régression | Tabulaire mixte | Très interprétable (règles) | Sur-apprend seul |
| Random Forest | Oui | Classif / Régression | Tabulaire mixte | Robuste, feature importance | Lent haute dimension |
| XGBoost / LightGBM | Oui | Classif / Régression | Tabulaire structuré | Meilleur sur tabulaire | Boîte noire relative |
| SVM | Oui | Classification | Texte TF-IDF, petits datasets | Excellent sur texte sparse | Lent sur grand dataset |
| Naive Bayes | Oui | Classification | Texte, features indépendantes | Ultra-rapide, bon NLP | Hypothèse d'indépendance |
| KNN | Oui | Classif / Régression | Petits datasets, faible dim. | Aucun entraînement | Lent inférence, haute dim. |
| MLP | Oui | Classif / Régression | Tabulaire > 10k samples | Non-linéaire, flexible | Sur-apprend petits datasets |
| CNN | Oui | Classif / Détection | Images, texte (n-grammes) | Excellent sur images | Requiert beaucoup de données |
| LSTM / GRU | Oui | Séquences | Texte séquentiel, séries temp. | Mémoire longue | Lent, instable |
| Transformers / BERT | Oui | NLP | Texte riche en contexte | État de l'art NLP | Très lourd, GPU requis |
| Autoencoder | Non | Représentation | Images, tabulaire | Détection anomalies | Évaluation difficile |
| GAN | Non | Génération | Images | Génération réaliste | Très difficile à entraîner |
| K-Means | Non | Clustering | Numérique, faible dim. | Simple, rapide | K fixé, clusters sphériques |
| DBSCAN | Non | Clustering | Formes arbitraires | Détecte les outliers | Sensible aux paramètres |
| Clustering Hiérarchique | Non | Clustering | Petits datasets | Visualisation complète | O(n²) mémoire |
| Isolation Forest | Non | Anomalies | Tabulaire | Fonctionne sans labels | Anomalies subjectives |
| LOF | Non | Anomalies | Clusters de densité variable | Anomalies contextuelles | Lent sur grand dataset |
| PCA | Non | Réduction dim. | Numérique dense | Rapide, débruitage | Linéaire |
| t-SNE / UMAP | Non | Visualisation | Haute dimension | Visualisation claire | t-SNE non réutilisable |
| Renforcement | — | Décision séq. | Environnement simulable | Optimal dynamique | Pas pour datasets statiques |

---

## 13. Données mixtes — numérique + texte

> C'est exactement le cas de ton dataset : des colonnes de constantes vitales (nombres) **et** une colonne de description des symptômes (texte libre).

---

### Le problème fondamental

Un modèle ne comprend **que des nombres**. Il ne peut pas lire "douleur thoracique intense". Il faut donc transformer le texte en nombres **avant** de le donner au modèle.

```
  Ce que tu as :
  ┌──────┬─────────┬─────────┬───────────────────────────────────┐
  │ age  │ tension │ sat_O2  │ description_symptomes             │
  ├──────┼─────────┼─────────┼───────────────────────────────────┤
  │  72  │   185   │   88    │ "douleur thoracique intense"       │
  │  35  │   120   │   97    │ "légère fatigue depuis ce matin"  │
  │  58  │   145   │   93    │ "essoufflement et palpitations"   │
  └──────┴─────────┴─────────┴───────────────────────────────────┘
        ↑                              ↑
   déjà des nombres            le modèle ne peut pas lire ça
```

---

### Étape 1 — Transformer le texte en nombres : la vectorisation

#### TF-IDF (la méthode classique)

TF-IDF crée **une colonne par mot** du vocabulaire. La valeur dans chaque colonne représente l'importance de ce mot dans la phrase.

```
  Vocabulaire détecté : [douleur, thoracique, intense, fatigue, légère,
                         essoufflement, palpitations, matin, ...]

  Phrase 1 : "douleur thoracique intense"
  → [douleur=0.6, thoracique=0.8, intense=0.5, fatigue=0, légère=0, ...]

  Phrase 2 : "légère fatigue depuis ce matin"
  → [douleur=0,   thoracique=0,   intense=0,   fatigue=0.7, légère=0.6, ...]
```

Résultat : chaque phrase devient une **ligne de nombres**. Si ton vocabulaire contient 5 000 mots distincts, tu obtiens 5 000 nouvelles colonnes. La plupart sont à zéro (matrice "sparse").

**TF = Term Frequency** : le mot "douleur" apparaît souvent dans cette phrase → valeur haute
**IDF = Inverse Document Frequency** : le mot "le" apparaît dans toutes les phrases → valeur basse (il n'est pas discriminant)

```
  TF-IDF pénalise les mots courants et valorise les mots rares et spécifiques :

  Mot "douleur"     → TF-IDF élevé   ← spécifique, discriminant
  Mot "le"          → TF-IDF = 0     ← dans toutes les phrases, inutile
  Mot "thoracique"  → TF-IDF très élevé ← rare et très spécifique
```

#### Embeddings (méthode avancée)

Au lieu de compter les mots, on représente chaque phrase par un vecteur dense de taille fixe (ex: 384 nombres) qui capture le **sens** de la phrase.

```
  "douleur thoracique intense"    → [0.82, -0.31, 0.15, 0.67, ...]  384 valeurs
  "forte douleur à la poitrine"   → [0.80, -0.29, 0.18, 0.65, ...]  384 valeurs
                                      ↑ presque identiques !
                                      → ces deux phrases veulent dire la même chose

  TF-IDF n'aurait pas fait ce lien (mots différents)
  L'embedding comprend le sens
```

---

### Étape 2 — Concaténer les deux types de features

Une fois le texte transformé en nombres, on **colle** les colonnes numériques et les colonnes texte côte à côte pour former une seule ligne par patient.

```
  Avant :
  ┌──────┬─────────┬─────────┐   +   ┌──────────┬─────────────┬────────┐
  │ age  │ tension │ sat_O2  │       │ douleur  │ thoracique  │  ...   │
  ├──────┼─────────┼─────────┤       ├──────────┼─────────────┼────────┤
  │  72  │   185   │   88    │       │   0.6    │     0.8     │  ...   │
  └──────┴─────────┴─────────┘       └──────────┴─────────────┴────────┘
   3 colonnes numériques               5000 colonnes TF-IDF

  Après concaténation :
  ┌──────┬─────────┬─────────┬──────────┬─────────────┬────────┐
  │ age  │ tension │ sat_O2  │ douleur  │ thoracique  │  ...   │
  ├──────┼─────────┼─────────┼──────────┼─────────────┼────────┤
  │  72  │   185   │   88    │   0.6    │     0.8     │  ...   │
  └──────┴─────────┴─────────┴──────────┴─────────────┴────────┘
                    5003 colonnes au total → 1 seule ligne par patient
```

Chaque patient est maintenant un **point dans un espace à 5003 dimensions**. Le modèle peut travailler dessus comme avec n'importe quel tableau.

---

### Étape 3 — Quel modèle utiliser sur ce mélange ?

Tous les modèles ne gèrent pas bien les matrices sparse (beaucoup de zéros) produites par TF-IDF :

```
  ┌─────────────────────┬────────────────────────────────────────────────────┐
  │ Modèle              │ Compatibilité avec données mixtes (num + TF-IDF)   │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ Régression          │ ✓ Excellent — conçu pour les matrices sparse        │
  │ Logistique          │                                                    │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ LinearSVC           │ ✓ Excellent — très rapide sur sparse               │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ Naive Bayes         │ ✓ Bon — conçu pour le texte                        │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ Random Forest       │ △ Moyen — lent sur 5000 colonnes, mais fonctionne  │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ XGBoost / LightGBM  │ △ Moyen — nécessite de convertir en dense ou       │
  │                     │   d'utiliser le format sparse natif de LightGBM    │
  ├─────────────────────┼────────────────────────────────────────────────────┤
  │ KNN                 │ ✗ Mauvais — malédiction de la dimensionnalité       │
  └─────────────────────┴────────────────────────────────────────────────────┘
```

---

### Exemple complet appliqué à ton dataset

```python
  import pandas as pd
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.preprocessing import StandardScaler
  import scipy.sparse as sp

  df = pd.read_csv("dataset_telemed.csv")

  # --- Colonnes numériques ---
  num_cols = ["age", "freq_cardiaque", "tension_sys", "temp", "sat_oxygene",
              "antecedents", "duree_symptomes"]
  X_num = StandardScaler().fit_transform(df[num_cols])
  # Shape : (2000, 7)  → 2000 patients, 7 colonnes numériques

  # --- Colonne texte ---
  tfidf = TfidfVectorizer(max_features=500)   # garder les 500 mots les plus utiles
  X_text = tfidf.fit_transform(df["description_symptomes"])
  # Shape : (2000, 500)  → 2000 patients, 500 colonnes-mots

  # --- Concaténation ---
  X_num_sparse = sp.csr_matrix(X_num)         # convertir en sparse pour coller
  X_final = sp.hstack([X_num_sparse, X_text]) # coller côte à côte
  # Shape : (2000, 507)  → 2000 patients, 507 colonnes au total

  y = df["niveau_urgence"]

  # --- Modèle ---
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression(class_weight={0:1, 1:2, 2:10})
  model.fit(X_final, y)
```

---

### Ce que ça donne sur un graphique (après PCA)

Avec 507 colonnes tu ne peux pas visualiser directement. En réduisant à 2D avec PCA :

```
  Composante 2 (différence entre types de cas)
      │
    2 │   ●●●●●
      │  ●●●●●●●         → les ● (non urgents) se regroupent
    0 │          ▲▲▲▲       les ▲ (urgences relatives) forment un groupe
      │         ▲▲▲▲▲▲      les ★ (urgences vitales) sont bien séparés
   -2 │               ★★★★
      │              ★★★★★
      └──────────────────────→ Composante 1 (état général de gravité)

  Si les groupes sont bien séparés visuellement → le modèle aura
  de bonnes performances (les classes sont naturellement distinctes)

  Si les groupes se mélangent → tâche difficile, il faudra
  soigner le class_weight pour les cas critiques
```

---

### Les deux pièges à éviter

**Piège 1 — Ne pas normaliser les numériques**
```
  age=72,  tension=185  →  la tension domine car ses valeurs sont plus grandes
  Après StandardScaler : age=1.2, tension=1.8  →  même échelle, même influence
```

**Piège 2 — Trop de features texte**
```
  TfidfVectorizer()                   → peut créer 50 000 colonnes
  TfidfVectorizer(max_features=500)   → limite à 500 mots les plus utiles
                                         beaucoup plus rapide, souvent aussi bon
```

---

## 14. Comment choisir son modèle — procédure pas à pas

> Suis les étapes dans l'ordre. À chaque étape, une ou deux questions suffisent à éliminer la majorité des modèles.

---

### Étape 1 — Est-ce que tu as des exemples étiquetés ?

```
  Tu as un dataset avec une colonne "réponse correcte" ?
  (ex: niveau_urgence, prix, spam/pas spam)
              │
      ┌───────┴───────┐
     OUI              NON
      │                │
  Apprentissage    Apprentissage
  SUPERVISÉ        NON SUPERVISÉ
  → Étape 2        → Étape 6
```

---

### Étape 2 — Que veux-tu prédire ? (supervisé)

```
  Ta colonne cible contient...
              │
   ┌──────────┼──────────┐
   │          │          │
Nombre     Catégorie   Rang /
continu    (classes)   score
(prix,     (urgent/    (1er, 2e...)
 durée)     non)
   │          │          │
Régression Classification  Régression
→ Étape 3  → Étape 3    (traité comme
                         régression)
```

---

### Étape 3 — Quel type de données as-tu ?

```
  Tes données ressemblent à quoi ?
                    │
      ┌─────────────┼─────────────┬──────────────┐
      │             │             │              │
  Tableau       Texte brut     Images         Séquence
  (colonnes     (phrases,      (photos,       ordonnée
  numériques    descriptions)  pixels)        dans le temps
  / catégories)                               (ECG, météo)
      │             │             │              │
  Étape 4       Étape 5a      → CNN           → LSTM / GRU
                               ou             ou Transformer
                               Transformer
```

---

### Étape 4 — Données tabulaires : combien d'exemples et quelle contrainte ?

```
  Nombre de lignes dans ton dataset
                │
    ┌───────────┼────────────┐
    │           │            │
  < 1 000    1 000 à      > 100 000
  exemples   100 000      exemples
    │        exemples         │
    │           │             │
  Modèles    Modèles       MLP ou
  simples    à arbres      XGBoost
  (LR, SVM,  → Étape 4b    avec
  RF)                      batches

  ┌────────────────────────────────────────────────────┐
  │ Étape 4b — contrainte métier ?                     │
  │                                                    │
  │  Tu dois expliquer les décisions à un humain ?     │
  │  (médecin, juge, client)                           │
  │        │                                           │
  │   ┌────┴────┐                                      │
  │  OUI       NON                                     │
  │   │         │                                      │
  │  Arbre    XGBoost / LightGBM  ← meilleur en général│
  │  de       (si performance max)│                    │
  │  Décision  Random Forest       │                   │
  │  ou        (si robustesse)     │                   │
  │  Régression                    │                   │
  │  Logistique                                        │
  └────────────────────────────────────────────────────┘

  Certaines erreurs sont bien pires que d'autres ?
  (ex: rater une urgence vitale = catastrophique)
        │
       OUI → utilise class_weight ou sample_weight
             avec n'importe lequel des modèles ci-dessus
             → voir Étape 7
```

---

### Étape 5a — Données texte : quelle approche ?

```
  Ton texte est...
              │
    ┌─────────┴──────────┐
    │                    │
  Court / mots-clés    Long / contexte
  (quelques mots,      important
  fréquence compte)    ("il ne souffre PAS")
    │                    │
  TF-IDF + modèle      Transformer
  classique            (BERT, CamemBERT)
    │
    ├── Régression Logistique   ← baseline rapide
    ├── LinearSVC               ← souvent meilleur
    └── Naive Bayes             ← très rapide, bon baseline
```

---

### Étape 5b — Données mixtes (tabulaire + texte) ?

```
  Tu as des colonnes numériques ET du texte ?
              │
             OUI
              │
  Option A (simple) :
    TF-IDF sur le texte → concaténer avec les colonnes numériques
    → XGBoost / Random Forest / Régression Logistique

  Option B (avancée) :
    Deux branches séparées fusionnées dans un MLP
    (une branche pour le tabulaire, une pour les embeddings texte)
    → nécessite plus de données et plus de code
```

---

### Étape 6 — Apprentissage non supervisé : quel besoin ?

```
  Qu'est-ce que tu veux faire ?
                    │
      ┌─────────────┼────────────────┐
      │             │                │
  Trouver des   Trouver des      Réduire le
  groupes       cas anormaux     nombre de
  similaires    (fraude, pannes) colonnes
      │             │                │
  Étape 6a      Étape 6b         Étape 6c

  Étape 6a — Clustering :
    Tu connais le nombre de groupes ?
    ├── OUI → K-Means
    └── NON → DBSCAN (si densité uniforme)
              ou Hiérarchique (si petit dataset + visualisation)

  Étape 6b — Anomalies sans labels :
    → Isolation Forest (tabulaire, rapide)
    → LOF (si les anomalies dépendent du contexte local)
    → Autoencoder (si les données sont complexes)

  Étape 6c — Réduction de dimensions :
    Besoin de visualiser ?
    ├── OUI → t-SNE ou UMAP (2D/3D uniquement)
    └── NON → PCA (preprocessing avant un modèle)
```

---

### Étape 7 — Gérer les erreurs coûteuses (coût asymétrique)

> Cette étape s'applique quand certaines erreurs sont bien plus graves que d'autres.
> Exemple : rater un cancer, rater une fraude, rater une urgence vitale.

```
  Problème : le modèle optimise l'accuracy globale par défaut
  → il peut ignorer la classe rare si elle est minoritaire

  Solution 1 — class_weight (le plus simple)
  ┌────────────────────────────────────────────────────┐
  │  sklearn : class_weight={0: 1, 1: 2, 2: 10}       │
  │  → la classe 2 (urgence vitale) pèse 10× plus      │
  │    dans le calcul de l'erreur                      │
  │  Disponible sur : LogisticRegression, RandomForest,│
  │  SVM, DecisionTree, MLP                            │
  └────────────────────────────────────────────────────┘

  Solution 2 — sample_weight (plus flexible)
  ┌────────────────────────────────────────────────────┐
  │  from sklearn.utils.class_weight import            │
  │      compute_sample_weight                         │
  │  w = compute_sample_weight({0:1, 1:2, 2:10}, y)   │
  │  model.fit(X, y, sample_weight=w)                  │
  │  Disponible sur : XGBoost, LightGBM                │
  └────────────────────────────────────────────────────┘

  Solution 3 — choisir la bonne métrique d'évaluation
  ┌────────────────────────────────────────────────────┐
  │  Ne pas utiliser l'accuracy seule                  │
  │                                                    │
  │  Recall classe 2 = TP2 / (TP2 + FN2)              │
  │    → "sur tous les vrais urgences vitales,         │
  │       combien ai-je détectées ?"                   │
  │    → c'est la métrique à maximiser en priorité     │
  │                                                    │
  │  F1-score pondéré = équilibre global               │
  │  Matrice de confusion = voir QUELLES erreurs       │
  │                         tu commets                 │
  └────────────────────────────────────────────────────┘
```

---

### Résumé visuel — arbre de décision global

```
  MON PROBLÈME
       │
       ├─ J'ai des labels ? ──NON──→ Clustering / Anomalies / Réduction dim.
       │
      OUI
       │
       ├─ Je prédis un nombre ? ──OUI──→ Régression Linéaire / Ridge / XGBoost
       │
       ├─ Je prédis une classe ?
       │        │
       │        ├─ Données tabulaires
       │        │        ├─ < 1 000 lignes  → Régression Logistique / SVM / RF
       │        │        ├─ 1k–100k lignes  → XGBoost / LightGBM / Random Forest
       │        │        └─ > 100k lignes   → XGBoost / MLP
       │        │
       │        ├─ Données texte
       │        │        ├─ Mots-clés / court  → TF-IDF + LogReg / LinearSVC
       │        │        └─ Contexte / long    → BERT / CamemBERT
       │        │
       │        ├─ Images                       → CNN / Vision Transformer
       │        │
       │        └─ Mixte (tabulaire + texte)
       │                 └─ TF-IDF concaténé   → XGBoost / Random Forest
       │
       └─ Erreurs coûteuses ? ──OUI──→ Ajouter class_weight ou sample_weight
                                        Évaluer avec Recall classe critique
```

---

### Appliqué à ton projet (dataset_telemed.csv)

```
  Étape 1 : OUI — on a niveau_urgence (0/1/2)
  Étape 2 : Classification multi-classe (3 classes)
  Étape 3 : Mixte → tabulaire (constantes vitales) + texte (description)
  Étape 4 : 2 000 lignes → modèles à arbres + option TF-IDF
  Étape 5b: Mixte → TF-IDF sur description + concat avec tabulaire
  Étape 7 : OUI — rater un niveau 2 est catastrophique
             → class_weight = {0: 1, 1: 2, 2: 10} (à affiner)
             → métrique prioritaire : Recall de la classe 2

  Ordre de test recommandé :
  ┌───┬──────────────────────────────┬────────────────────────────────┐
  │ # │ Modèle                       │ Pourquoi                       │
  ├───┼──────────────────────────────┼────────────────────────────────┤
  │ 1 │ Régression Logistique        │ Baseline rapide, interprétable │
  │ 2 │ Random Forest                │ Robuste, donne feature import. │
  │ 3 │ LightGBM                     │ Probablement le meilleur       │
  │ 4 │ XGBoost                      │ Comparaison avec LightGBM      │
  │ 5 │ MLP (si les autres plafonnent)│ Dernier recours               │
  └───┴──────────────────────────────┴────────────────────────────────┘

  Pour chaque modèle :
  1. Entraîner AVEC class_weight
  2. Évaluer avec : Recall classe 2, F1 pondéré, matrice de confusion
  3. Logger dans MLflow
  4. Comparer → choisir le meilleur rapport performance / interprétabilité
```
