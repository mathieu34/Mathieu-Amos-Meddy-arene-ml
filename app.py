
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
 
# ─────────────────────────────────────────────
# Chargement du modèle
# ─────────────────────────────────────────────
MODEL_PATH = "champion.joblib"
 
@st.cache_resource
def charger_modele():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)
 
modele = charger_modele()
 
# Features attendues par le modèle (ordre exact)
FEATURES = [
    'age', 'balance', 'day_of_week', 'duration', 'campaign',
    'pdays', 'previous',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
    'job_management', 'job_retired', 'job_self-employed',
    'job_services', 'job_student', 'job_technician', 'job_unemployed',
    'marital_married', 'marital_single',
    'education_secondary', 'education_tertiary',
    'default_yes', 'housing_yes', 'loan_yes',
    'contact_telephone',
    'month_aug', 'month_dec', 'month_feb', 'month_jan', 'month_jul',
    'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
    'month_sep',
    'poutcome_other', 'poutcome_success'
]
 
# ─────────────────────────────────────────────
# Interface
# ─────────────────────────────────────────────
st.title("Prédiction de souscription bancaire")
st.markdown(
    "Renseignez le profil du client pour estimer sa probabilité "
    "de souscrire un dépôt à terme."
)
 
if modele is None:
    st.warning(
        "Modèle non disponible — `champion.joblib` introuvable. "
        "Placez le fichier dans le même dossier que `app.py` et relancez."
    )
    st.stop()
 
st.markdown("---")
 
# ─────────────────────────────────────────────
# Formulaire client
# ─────────────────────────────────────────────
st.subheader("Profil du client")
 
col1, col2 = st.columns(2)
 
with col1:
    age = st.number_input("Âge", min_value=18, max_value=95, value=40, step=1)
 
    job = st.selectbox("Profession", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
 
    marital = st.selectbox(
        "Situation familiale", ["married", "single", "divorced"]
    )
 
    education = st.selectbox(
        "Niveau d'éducation", ["primary", "secondary", "tertiary", "unknown"]
    )
 
    default = st.selectbox("Défaut de crédit ?", ["no", "yes"])
    housing = st.selectbox("Crédit immobilier ?",  ["no", "yes"])
    loan    = st.selectbox("Crédit personnel ?",   ["no", "yes"])
 
with col2:
    balance = st.number_input(
        "Solde annuel moyen (€)",
        min_value=-10000, max_value=100000, value=1500, step=100
    )
 
    contact = st.selectbox(
        "Type de contact", ["cellular", "telephone", "unknown"]
    )
 
    month = st.selectbox("Mois du dernier contact", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
 
    day_of_week = st.number_input(
        "Jour du mois", min_value=1, max_value=31, value=15, step=1
    )
 
    duration = st.number_input(
        "Durée dernier appel (sec.)",
        min_value=0, max_value=5000, value=200, step=10
    )
 
    campaign = st.number_input(
        "Nb contacts cette campagne", min_value=1, max_value=50, value=2, step=1
    )
 
    pdays = st.number_input(
        "Jours depuis dernier contact (-1 = jamais)",
        min_value=-1, max_value=999, value=-1, step=1
    )
 
    previous = st.number_input(
        "Nb contacts campagnes précédentes",
        min_value=0, max_value=50, value=0, step=1
    )
 
    poutcome = st.selectbox(
        "Résultat campagne précédente",
        ["unknown", "other", "failure", "success"]
    )
 
st.caption(
    "ℹ️ `duration` est ultra-prédictive mais inconnue avant l'appel. "
    "En prod, utilisez avec précaution."
)
 
# ─────────────────────────────────────────────
# Encodage
# ─────────────────────────────────────────────
def encoder_client(age, balance, day_of_week, duration, campaign,
                   pdays, previous, job, marital, education,
                   default, housing, loan, contact, month, poutcome):
    """Encode le profil client en vecteur de 38 features.
    Correspond exactement au preprocessing du champion.joblib.
    """
    row = {f: 0 for f in FEATURES}
 
    # numériques directs
    row['age']         = age
    row['balance']     = balance
    row['day_of_week'] = day_of_week
    row['duration']    = duration
    row['campaign']    = campaign
    row['pdays']       = pdays
    row['previous']    = previous
 
    # job — one-hot (référence : admin.)
    if job != "admin." and job != "unknown":
        col = f"job_{job}"
        if col in row:
            row[col] = 1
 
    # marital — one-hot (référence : divorced)
    if marital in ("married", "single"):
        row[f"marital_{marital}"] = 1
 
    # education — one-hot (référence : primary)
    if education in ("secondary", "tertiary"):
        row[f"education_{education}"] = 1
 
    # binaires
    row['default_yes'] = 1 if default == "yes" else 0
    row['housing_yes'] = 1 if housing == "yes" else 0
    row['loan_yes']    = 1 if loan    == "yes" else 0
 
    # contact — one-hot (référence : cellular)
    if contact == "telephone":
        row['contact_telephone'] = 1
 
    # month — one-hot (référence : apr)
    month_col = f"month_{month}"
    if month_col in row:
        row[month_col] = 1
 
    # poutcome — one-hot (référence : failure)
    if poutcome in ("other", "success"):
        row[f"poutcome_{poutcome}"] = 1
 
    return np.array([row[f] for f in FEATURES], dtype=float)
 
# ─────────────────────────────────────────────
# Prédiction
# ─────────────────────────────────────────────
if st.button("Prédire"):
 
    # validations
    if duration < 0:
        st.error("La durée d'appel ne peut pas être négative.")
        st.stop()
 
    if balance < -5000 or balance > 50000:
        st.warning(
            f"Solde de {balance}€ hors de la plage habituelle "
            "[-5000, 50000]. Prédiction moins fiable."
        )
 
    # encoder
    vecteur = encoder_client(
        age, balance, day_of_week, duration, campaign,
        pdays, previous, job, marital, education,
        default, housing, loan, contact, month, poutcome
    )
 
    # prédire — pas de scaler (le modèle n'en a pas besoin)
    X = vecteur.reshape(1, -1)
    pred  = modele.predict(X)[0]
    proba = modele.predict_proba(X)[0]
 
    # classes : [False, True] → True = souscrit
    idx_yes   = list(modele.classes_).index(True)
    proba_yes = proba[idx_yes]
 
    st.markdown("---")
    st.subheader("Résultat")
 
    if pred:
        st.success(f"**SOUSCRIRA** — probabilité : {proba_yes:.1%}")
        st.info("→ Recommandation : **Appeler en priorité**")
    else:
        st.error(f"**NE SOUSCRIRA PAS** — probabilité yes : {proba_yes:.1%}")
        st.info("→ Recommandation : **Passer au client suivant**")
 
    st.metric("Probabilité de souscription", f"{proba_yes:.1%}")
    st.progress(float(proba_yes))
 
    with st.expander("Détail du vecteur envoyé au modèle"):
        df_feat = pd.DataFrame({
            "Feature": FEATURES,
            "Valeur":  vecteur
        })
        st.dataframe(df_feat)
 
st.markdown("---")
st.caption(
    "Modèle : RandomForestClassifier | "
    "Dataset : Bank Marketing UCI | "
    "Métriques : ROC-AUC + Recall (yes)"
)