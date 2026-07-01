# Sujet d’Examen
## Diagnostic assisté et tri d'urgence multimodal en télémédecine

L'engorgement des services d'urgence et le développement de la télémédecine imposent des solutions de tri rapide et fiables.  
Ce projet consiste à concevoir un système d'intelligence artificielle capable de classer le degré d'urgence d'une situation à partir de données hybrides (numériques et textuelles).

Pour créer ce système d’IA, vous aurez à votre disposition un jeu de données composé de données collectées au sein d’un établissement médical.  
Il contient **2000 échantillons** pour lesquels nous connaissons le **niveau d’urgence** (décision prise par un professionnel de santé à la lecture des informations).

---

## Description des données

Dans le détail, chaque échantillon est composé de :

### 1. Données tabulaires
- Âge
- Données administratives
- Constantes vitales :
  - Fréquence cardiaque
  - Tension artérielle
  - Température
  - Saturation en oxygène
- Antécédents médicaux (pathologie chronique)
- Durée des symptômes

### 2. Données textuelles
- Description libre rédigée par le patient **ou**
- Rapport d'appel du régulateur détaillant la plainte principale

### 3. Variable cible
- **Niveau d'urgence** :
  - `0` : Non urgent
  - `1` : Urgence relative
  - `2` : Urgence vitale

---

## Enjeu du projet

L’enjeu est d’exploiter ces données afin de prédire si une situation relève :
- de l’urgence vitale,
- d’une urgence relative,
- ou d’une situation non urgente.

Vous devrez pour cela :
- Respecter les **contraintes réglementaires et éthiques** liées à l’usage de données personnelles sensibles
- Adapter vos méthodes d’apprentissage afin de **limiter les erreurs de classification dangereuses** d’un point de vue métier

---

## Objectif du projet

L’objectif est de prédire le degré d’urgence d’une demande entrante à partir des informations renseignées.

Il s’agit donc d’une **tâche supervisée de classification multi-classe (3 classes)**.

Vous devrez :

- Entraîner plusieurs modèles :
  - Random Forest
  - XGBoost
  - Réseaux de neurones
  - Autres modèles pertinents
- Comparer leurs performances
- Utiliser des métriques de classification adaptées :
  - Accuracy
  - F1-score pondéré
  - Matrice de confusion
- Mettre en place une **validation croisée**
- Discuter l’architecture de modèle présentant le meilleur **rapport performance / coût computationnel d’inférence**

---

## Gestion des erreurs critiques et réflexion éthique

Un enjeu éthique majeur devra être abordé :

> *Une situation d’urgence vitale (niveau 2) classée niveau 1 ou 0 est une erreur bien plus grave qu’une situation non urgente (niveau 0) classée niveau 1 ou 2.*

Concrètement, vous devrez :
- Modifier votre **meilleur modèle**
- Favoriser la **réduction des erreurs critiques**
- Identifier et justifier la **métrique à privilégier** lors de l’apprentissage

---

## Analyse comparée des scénarios

Vous comparerez les performances des modèles selon plusieurs scénarios d’entraînement afin d’évaluer l’impact des types de données utilisées.

### Scénarios étudiés

- **Scénario 1 – Approche multimodale complète**  
  Utilisation de l’ensemble des variables (données tabulaires + texte vectorisé)

- **Scénario 2 – Sans variables sensibles**  
  Retrait des variables éthiquement discutables ou sources de biais, avec justification argumentée

- **Scénario 3 – Diagnostic "aveugle" (NLP seul)**  
  Prédiction basée uniquement sur la description textuelle des symptômes

- **Scénario 4 – Données cliniques pures (Tabulaire seul)**  
  Prédiction basée uniquement sur les constantes vitales et l’âge

Votre objectif est de :
- Documenter l’impact de chaque scénario
- Comparer les performances
- Argumenter le choix du modèle et des données dans un **contexte réel**, en tenant compte des enjeux :
  - Éthiques
  - Légaux
  - De robustesse

---

## Industrialisation et Déploiement

Une fois les modèles validés, vous passerez à l’**industrialisation** de la solution IA via une application destinée à un utilisateur non technique.

### Architecture de la solution

L’application sera composée de :

#### 1. API de prédiction
- Service exposé via requêtes HTTP `POST`
- Chargement du modèle entraîné
- Routes minimales :
  - Prédiction du niveau d’urgence
  - Réentraînement monitoré
  - Vérification de l’état de santé de l’API
- Journalisation des requêtes :
  - Entrées
  - Sorties
  - Date d’inférence
  - Utilisateur ou ID de session
- Gestion rigoureuse :
  - Validation des données
  - Gestion des erreurs
  - Robustesse générale de l’API

#### 2. Interface utilisateur graphique
- Formulaire simple de saisie
- Affichage clair et interprétable de la prédiction
- Consultation de l’historique des inférences (si possible)

---

## Suivi, Monitoring et CI/CD

- Intégration de **MLflow** pour :
  - Suivi des versions de modèles
  - Hyperparamètres
  - Performances
  - Versionning contrôlé
- Monitoring de l’API :
  - Temps de réponse
  - Taux d’erreurs
  - Disponibilité (ex : Uptime Kuma)
  - Solutions possibles :
    - Prometheus + Grafana
    - Logs et alertes automatisées

### CI/CD

- Mise en place d’une chaîne CI/CD complète avec **GitHub Actions**
- Automatisation :
  - Tests
  - Déploiement
- À chaque mise à jour de la branche principale :
  - Construction automatique d’une image Docker
  - Déploiement sur l’infrastructure cible

Objectif : garantir un déploiement **reproductible**, **traçable** et conforme aux **bonnes pratiques d’ingénierie IA**.

---

## Éthique et cadre réglementaire

Une section spécifique devra traiter :

- La conformité au **RGPD** (données de santé sensibles)
- La responsabilité juridique en cas d’erreur de classification de l’urgence vitale

---

## Description détaillée des variables

| Colonne | Description |
|-------|------------|
| `patient_id` | Identifiant unique du patient |
| `sexe` | Sexe du patient : F (Femme) ou H (Homme) |
| `age` | Âge du patient (en années) |
| `zone_vie` | Type de résidence : U (Urbain) ou R (Rural) |
| `source` | Origine de la donnée : appel (transcription) ou chat |
| `freq_cardiaque` | Fréquence cardiaque (bpm) |
| `tension_sys` | Tension artérielle systolique (mmHg) |
| `temp` | Température corporelle (°C) |
| `sat_oxygene` | Saturation en oxygène (%) |
| `antecedents` | Pathologies chroniques : 1 (Oui), 0 (Non) |
| `duree_symptomes` | Durée des symptômes (en heures) |
| `description_symptomes` | Texte libre décrivant l’état du patient |
| `niveau_urgence` | Variable cible : 0 (Non urgent), 1 (Urgence relative), 2 (Urgence vitale) |