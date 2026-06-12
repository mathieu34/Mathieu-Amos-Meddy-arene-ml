# Mathieu-Amos-Meddy-arene-ml

Arène ML — Prédiction de souscription bancaire

Une banque fait des appels pour proposer un dépôt à terme. On veut prédire si un client va dire `yes` ou `no` — pour appeler en priorité ceux qui ont le plus de chances de souscrire.

**Problème :** classification binaire | **Cible :** colonne `y`

---

## Dataset

Bank Marketing – UCI · `bank-full.csv` · 45 211 lignes · 16 features + 1 cible · séparateur `;`

Mix catégoriel / numérique. `"unknown"` apparaît dans `job`, `education`, `contact`, `poutcome` — officiellement une catégorie à part entière selon UCI, mais qu'on peut choisir de traiter comme valeur manquante selon les résultats.

Cible déséquilibrée : 88% `no` / 12% `yes` → l'accuracy seule ment, un modèle qui prédit toujours `no` fait déjà 88% sans rien apprendre.

**Métriques retenues :** ROC-AUC + recall sur `yes`

**Split :** 80/20 avec `stratify=y` et `random_state=42`

---

## Preprocessing

- `unknown` dans `job`, `education`, `contact`, `poutcome` → catégorie officielle selon UCI, mais à tester : la garder telle quelle ou la traiter comme manquant (mode de la colonne)
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

## WebApp

Streamlit : on saisit le profil d'un client, l'app affiche la proba de souscription et une reco ("appeler" / "passer").

`champion.joblib` contient le modèle et le scaler — les deux vont ensemble, sinon la normalisation en prod ne colle pas avec celle du training.

```bash
pip install pandas scikit-learn streamlit joblib matplotlib seaborn
streamlit run app.py
```

---

## Répartition des rôles

| Rôle | Responsable |
|------|-------------|
| Chargement & nettoyage | Meddy |
| Arène (algos & comparaison) | Mathieu |
| WebApp Streamlit | Amos |
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
├── data/bank-full.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_arene.ipynb
├── app.py
├── champion.joblib
└── README.md
```
