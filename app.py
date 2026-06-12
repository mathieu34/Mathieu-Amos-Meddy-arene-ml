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
        return None, None
    bundle = joblib.load(MODEL_PATH)
    return bundle["modele"], bundle["scaler"]

modele, scaler = charger_modele()

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
        "Entraînez et sauvegardez le champion, puis relancez l'app."
    )
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────
# Formulaire client
# ─────────────────────────────────────────────
st.subheader("Profil du client")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Âge", min_value=18, max_value=95, value=40, step=1
    )
    job = st.selectbox(
        "Profession", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ]
    )
    marital = st.selectbox(
        "Situation familiale", ["married", "single", "divorced"]
    )
    education = st.selectbox(
        "Niveau d'éducation", ["primary", "secondary", "tertiary", "unknown"]
    )
    default = st.selectbox("Défaut de crédit ?", ["no", "yes"])

with col2:
    balance = st.number_input(
        "Solde annuel moyen (€)", min_value=-10000, max_value=100000,
        value=1500, step=100
    )
    housing = st.selectbox("Crédit immobilier ?", ["no", "yes"])
    loan    = st.selectbox("Crédit personnel ?",  ["no", "yes"])
    duration = st.number_input(
        "Durée dernier appel (sec.)", min_value=0, max_value=5000,
        value=200, step=10
    )
    campaign = st.number_input(
        "Nb contacts cette campagne", min_value=1, max_value=50,
        value=2, step=1
    )

st.caption(
    "`duration` est ultra-prédictive mais inconnue avant l'appel. "
    "Le modèle est entraîné avec — en prod, utilisez avec précaution."
)

# ─────────────────────────────────────────────
# Construction du vecteur de features
# ─────────────────────────────────────────────
def encoder_client(age, job, marital, education, default,
                   balance, housing, loan, duration, campaign):
    """Encode le profil client en vecteur numérique.
    
    """
    # binaires
    default_enc  = 1 if default  == "yes" else 0
    housing_enc  = 1 if housing  == "yes" else 0
    loan_enc     = 1 if loan     == "yes" else 0

    # ordinal education
    edu_map = {"primary": 0, "secondary": 1, "tertiary": 2, "unknown": 1}
    education_enc = edu_map[education]

    # one-hot job (12 modalités — ordre alphabétique)
    jobs = [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ]
    job_enc = [1 if job == j else 0 for j in jobs]

    # one-hot marital (3 modalités)
    maritals = ["divorced", "married", "single"]
    marital_enc = [1 if marital == m else 0 for m in maritals]

    # assembler le vecteur
    vecteur = np.array([
        age, balance, duration, campaign,
        default_enc, housing_enc, loan_enc, education_enc,
        *job_enc,
        *marital_enc
    ], dtype=float)

    return vecteur

# ─────────────────────────────────────────────
# Prédiction
# ─────────────────────────────────────────────
if st.button("Prédire"):

    # cas limite : durée négative ou nulle
    if duration < 0:
        st.error("La durée d'appel ne peut pas être négative.")
        st.stop()

    # cas limite : âge hors plage réaliste
    if age < 18 or age > 95:
        st.error("Âge hors plage (18–95 ans).")
        st.stop()

    # cas adversarial : balance aberrante
    if balance > 50000 or balance < -5000:
        st.warning(
            f"Solde de {balance}€ hors de la plage habituelle "
            "[-5000, 50000]. Prédiction moins fiable."
        )

    # encoder et normaliser
    vecteur = encoder_client(
        age, job, marital, education, default,
        balance, housing, loan, duration, campaign
    )

    try:
        X_scaled = scaler.transform(vecteur.reshape(1, -1))
        pred     = modele.predict(X_scaled)[0]
        proba    = modele.predict_proba(X_scaled)[0]

        proba_yes = proba[1] if len(proba) > 1 else proba[0]

        st.markdown("---")
        st.subheader("Résultat")

        if pred == 1:
            st.success(f"**SOUSCRIRA** — probabilité : {proba_yes:.1%}")
            st.info("→ Recommandation : **Appeler en priorité**")
        else:
            st.error(f"**NE SOUSCRIRA PAS** — probabilité yes : {proba_yes:.1%}")
            st.info("→ Recommandation : **Passer au client suivant**")

        st.metric("Probabilité de souscription", f"{proba_yes:.1%}")
        st.progress(float(proba_yes))

    except Exception as e:
        st.error(
            f"Erreur lors de la prédiction : {e}\n\n"
            "Vérifiez que le `champion.joblib` correspond bien "
            "au même preprocessing que cette app."
        )

st.markdown("---")
st.caption(
    "Modèle : champion.joblib | "
    "Dataset : Bank Marketing UCI | "
    "Métriques : ROC-AUC + Recall (yes)"
)
