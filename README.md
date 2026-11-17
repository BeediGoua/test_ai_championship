# AI Championship

Base de travail pour évaluer des scores binaires, explorer des expériences
reproductibles et produire un fichier de prédictions.

## Objectifs

- Calculer et tracer la courbe ROC, l'AUC, le taux de faux positifs (FPR) et le
  taux de vrais positifs (TPR), avec validation des entrées.
- Comparer des expériences exploratoires sur des données synthétiques ou des
  données de challenge.
- Générer un CSV de soumission en conservant l'ordre fourni des identifiants.
- Garder une trace des expériences dans `reports/experiment_log.csv`.

Les notebooks sont des exemples autonomes : ils ne présument ni d'un jeu de
données, ni d'une implémentation officielle de KGW ou TextSeal. Remplacez les
données synthétiques et les détecteurs jouets par les données et méthodes du
challenge lorsqu'elles seront disponibles.

## Prérequis et installation

Python 3.10 à 3.13 est recommandé (les dépendances figées ont été validées
avec Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Les versions des dépendances directes sont figées dans `requirements.txt`.

## Mode d'exécution

Depuis la racine du dépôt :

```bash
jupyter lab
```

Ouvrez et exécutez les notebooks dans l'ordre :

1. `experiments/01_kgw_metrics.ipynb` — exemple de scores de détection et
   métriques ROC.
2. `experiments/02_textseal_or_toy.ipynb` — comparaison de scores sur un
   exemple jouet.
3. `experiments/03_image_robustness.ipynb` — effet d'une perturbation
   synthétique sur des scores d'image.
4. `experiments/04_mock_challenge.ipynb` — flux de bout en bout avec export
   d'une soumission factice.

Les fonctions partagées sont dans `src/metrics.py` et `src/submission.py`.
Les notebooks enregistrent leurs fichiers générés dans `outputs/`. Le journal
des expériences est un CSV à compléter manuellement dans
`reports/experiment_log.csv`.

## Format de soumission

`src/submission.py` construit un CSV à deux colonnes, `id` et `prediction` par
défaut. Les identifiants et prédictions doivent être de même longueur ; les
identifiants doivent être uniques et les prédictions numériques et finies.
L'ordre des identifiants transmis est conservé. Adaptez les noms de colonnes
si le format officiel du challenge diffère.
