# BCS Cattle Predictor — Multi-vues flexible

Cette version accepte **1 à 4 vues** du même animal.

## Principe

Le modèle est entraîné sur les 4 vues standardisées présentes dans la base, mais pendant
l'entraînement certaines vues sont masquées aléatoirement (`view dropout`).

Il apprend donc à prédire avec :
- 4 vues
- 3 vues
- 2 vues
- 1 vue

Les performances sont évaluées séparément pour chaque nombre de vues.

## Vues standardisées

1. Dorso-caudale
2. Arrière
3. Latérale gauche
4. Latérale droite

## Secrets

Créer :

`.streamlit/secrets.toml`

avec :

```toml
SUPABASE_URL = "https://xxxxxxxxxxxx.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_xxxxxxxxxxxxx"
```

## Installation

```bat
python -m pip install -r requirements.txt
```

## Entraînement

```bat
python train_multiview_flexible.py
```

Le modèle sera stocké sous :

`bcs-models/models/bcs_mobilenet_v3_flexible.pt`

## Lancer l'application

```bat
python -m streamlit run app.py
```

L'application accepte de 1 à 4 photos, chacune dans son emplacement correspondant à la vue.
Elle affiche également les performances de validation correspondant au nombre de vues fourni.
