# Télémédecine — Assistant de tri d'urgence

Application d'**aide au tri d'urgence** pour la télémédecine. À partir des **constantes vitales**
d'un patient et d'une **description libre de ses symptômes**, l'assistant estime un **niveau
d'urgence** pour aider à prioriser la prise en charge :

| Niveau | Signification |
|:---:|---|
| 🟢 **0** | **Pas urgent** — situation bénigne |
| 🟠 **1** | **Urgent** — à prendre en charge rapidement |
| 🔴 **2** | **Très urgent** — urgence vitale, priorité absolue |

> ⚠️ **Outil d'aide à la décision, pas un dispositif médical.** La prédiction ne remplace jamais
> le jugement d'un professionnel de santé : la décision finale reste **humaine**. L'assistant est
> volontairement **prudent** — en cas de doute, il préfère surestimer l'urgence plutôt que de
> risquer de passer à côté d'un cas grave.

---

## Démarrage rapide

L'application se lance via **Docker** (Docker + Docker Compose requis). Les images sont **déjà
publiées** et contiennent tout le nécessaire (code + modèle) : il suffit de les **télécharger**,
aucune compilation n'est requise.

```bash
docker compose pull      # télécharge les images prêtes à l'emploi
docker compose up -d     # démarre l'application
```

Une fois démarrée, ouvrez l'interface dans votre navigateur :

👉 **http://localhost:8501**

Pour arrêter l'application :

```bash
docker compose down
```

---

## Utiliser l'interface

1. **Renseignez les informations du patient** dans le formulaire :
   - constantes vitales : âge, fréquence cardiaque, tension, température, saturation en oxygène…
   - une **description des symptômes** en texte libre (ex. *« douleur thoracique intense depuis
     ce matin »*).
2. Cliquez sur **« Prédire le niveau d'urgence »**.
3. L'assistant affiche :
   - le **niveau d'urgence** prédit (couleur + libellé),
   - les **probabilités** pour chacun des 3 niveaux,
   - un **historique** des prédictions précédentes.

### Alertes de sécurité

Indépendamment de la prédiction, l'interface **signale automatiquement** les valeurs
physiologiquement critiques (ex. saturation en oxygène < 80 %, fréquence cardiaque très anormale,
température extrême). Ces alertes s'affichent **même si le modèle ne prédit pas « très urgent »**,
comme filet de sécurité.

---

## Utiliser l'API (intégrateurs)

L'application expose une API. La documentation interactive est disponible sur
**http://localhost:8000/docs**.

Exemple de prédiction :

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 62,
    "freq_cardiaque": 130,
    "tension_sys": 180,
    "temp": 39.5,
    "sat_oxygene": 86,
    "antecedents": 1,
    "duree_symptomes": 2.0,
    "source": "appel",
    "description_symptomes": "détresse respiratoire et douleur thoracique"
  }'
```

Réponse :

```json
{
  "prediction": 2,
  "label": "Très urgent",
  "probabilites": { "Pas urgent": 0.02, "Urgent": 0.11, "Très urgent": 0.87 },
  "model_name": "RandomForest",
  "timestamp": "..."
}
```

Principaux points d'entrée :

| Endpoint | Rôle |
|---|---|
| `POST /predict` | prédire le niveau d'urgence d'un patient |
| `GET /health` | état de l'API |
| `GET /history` | historique des prédictions |
| `GET /docs` | documentation interactive (Swagger) |

---

## Ce qui tourne en arrière-plan

`docker compose up` démarre également un **tableau de bord de supervision** (facultatif pour
l'utilisation, utile pour la surveillance) :

| Outil | Adresse | Rôle |
|---|---|---|
| Grafana | http://localhost:3000 | tableaux de bord de suivi (identifiants : `admin` / `admin`) |
| Prometheus | http://localhost:9090 | collecte des métriques |
| Uptime-Kuma | http://localhost:3001 | surveillance de disponibilité |
| MLflow | http://localhost:5000 | suivi des modèles |

---

## Bon à savoir

- L'assistant s'appuie sur un modèle entraîné à partir de **données historiques** ; ses prédictions
  reflètent ces données et peuvent se tromper, notamment sur des cas cliniques atypiques.
- Aucune donnée sensible de discrimination n'est utilisée (le modèle n'exploite ni le sexe ni le lieu de vie du patient).
- En cas d'urgence réelle, **contactez les services d'urgence** — cet outil ne s'y substitue pas.
