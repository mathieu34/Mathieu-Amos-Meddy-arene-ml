# Mathieu-Amos-Meddy-arene-ml

Arène ML — Prédiction de souscription bancaire

Une banque fait des appels pour proposer un dépôt à terme. On veut prédire si un client va dire `yes` ou `no` — pour appeler en priorité ceux qui ont le plus de chances de souscrire.

**Problème :** classification binaire | **Cible :** colonne `y`

---

## Dataset

Bank Marketing – UCI · 45 211 lignes · 16 features + 1 cible

Mix catégoriel / numérique. `"unknown"` apparaît dans `job`, `education`, `contact`, `poutcome` — officiellement une catégorie à part entière selon UCI, mais qu'on peut choisir de traiter comme valeur manquante selon les résultats.

Cible déséquilibrée : 88% `no` / 12% `yes` → l'accuracy seule ment, un modèle qui prédit toujours `no` fait déjà 88% sans rien apprendre.

**Métriques retenues :** ROC-AUC + recall sur `yes`

**Split :** 80/20 avec `stratify=y` et `random_state=42`

---

## Preprocessing

- `unknown` dans `job`, `education`, `contact`, `poutcome` → catégorie officielle selon UCI, conservée telle quelle
- **Encodage :** One-Hot pour les nominales (`job`, `marital`…), Ordinal pour `education` (primary < secondary < tertiary), binaire 0/1 pour `default`, `housing`, `loan`
- `month` : ordinal ou one-hot ? À décider après exploration
- **Scaling :** `StandardScaler` ajusté sur le train uniquement — jamais sur le jeu complet avant le split

> Règle d'or : tout ce qui s'apprend sur la donnée (médiane, scaler, catégories) se calcule sur le train et s'applique au test. Le `Pipeline` scikit-learn rend ça impossible à rater.

---

## La feature `duration`

La durée du dernier appel est ultra-prédictive… mais on ne la connaît pas avant d'appeler. En prod, elle est inutilisable. On entraîne avec et sans, on mesure l'écart, et on réfléchit à ce que ça dit sur le modèle.

---

## Les algos (l'Arène)

| Algo | Rôle |
|------|------|
| Régression logistique | Baseline — interprétable, rapide |
| Random Forest | Robuste, peu de réglages, gère bien le mix |
| Gradient Boosting | Souvent le meilleur sur tabulaire |

Même split, même métrique pour tous.

---

## Évaluation

- Cross-validation 5-fold pour fiabiliser le classement
- Matrice de confusion + courbe ROC sur le champion
- Baseline : prédire toujours `no` → recall `yes` = 0%

Rater un souscripteur (faux négatif) ≠ appeler un non-souscripteur pour rien (faux positif) → le seuil de décision est un choix métier.

### Leaderboard

| Algo | ROC-AUC | Recall (yes) | Temps | Commentaire |
|------|---------|--------------|-------|-------------|
| Régression logistique | — | — | — | baseline |
| Random Forest | — | — | — | |
| Gradient Boosting | — | — | — | |

---

## Pipeline complet — comment ça s'enchaîne

```
eda.ipynb
  └─ charge les données depuis UCI (fetch_ucirepo)
  └─ explore, nettoie, documente les choix
  └─ exporte data/bank_processed.csv
          │
          ▼
arene.ipynb
  └─ charge data/bank_processed.csv
  └─ encode, split, entraîne 3 algos, compare
  └─ sauvegarde le champion → champion.joblib
          │
          ▼
app.py (WebApp Streamlit)
  └─ charge champion.joblib une seule fois au démarrage
  └─ reçoit le profil d'un client via le formulaire
  └─ prédit et affiche la proba de souscription
```

---

## Lancer les notebooks

```bash
pip install pandas scikit-learn streamlit joblib matplotlib seaborn ucimlrepo jupyter
```

> Les commandes utilisent `python -m jupyter` pour forcer l'exécution dans le bon environnement virtuel.  
> Remplacer `TON_VENV` par le nom de ton environnement (ex: `Ipssi_env`).

**Enregistrer le kernel du venv (une seule fois) :**

```bash
python -m ipykernel install --user --name=TON_VENV
```

**Tout lancer d'un coup (dans l'ordre) :**

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=TON_VENV notebooks/eda.ipynb && python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=TON_VENV notebooks/arene.ipynb
```

**WebApp (nécessite `champion.joblib` généré à l'étape précédente) :**

```bash
streamlit run app.py
```

---

## Répartition des rôles

| Rôle | Responsable |
|------|-------------|
| Chargement & nettoyage (`eda.ipynb`) | Meddy |
| Arène — algos & comparaison (`arene.ipynb`) | Mathieu |
| WebApp Streamlit (`app.py`) | Amos |
| README & soutenance | Meddy + Mathieu + Amos |

---

## Questions ouvertes

- `month` : ordinal, one-hot ou cyclique (sin/cos) ?
- `duration` : avec ou sans ? Comparer les deux
- Classes déséquilibrées : sous-échantillonner ou `class_weight='balanced'` ?
- `unknown` : garder comme catégorie ou remplacer par le mode ?

---

## Structure

```
.
├── data/
│   ├── bank_processed.csv                ← généré par eda.ipynb (avec duration)
│   └── bank_processed_no_duration.csv    ← généré par eda.ipynb (sans duration)
├── notebooks/
│   ├── eda.ipynb                         ← chargement UCI, nettoyage, export CSV
│   └── arene.ipynb                       ← modélisation, leaderboard, export champion
├── app.py                                ← WebApp Streamlit
├── champion.joblib                       ← généré par arene.ipynb, chargé par app.py
└── README.md
```
