"""Lecture du fichier et préparation des variables numériques."""
import numpy as np
import pandas as pd
import streamlit as st


def load_numeric(uploaded_file):
    """Lit un CSV ou un Excel et renvoie (df, X, colonnes numériques) ou None si inutilisable."""
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    num = df.select_dtypes(include=np.number)
    num = num.loc[:, num.notna().all()]          # colonnes sans valeur manquante
    if num.shape[1] < 2:
        st.error("Le fichier doit contenir au moins deux colonnes numériques complètes.")
        return None
    ignored = [c for c in df.columns if c not in num.columns]
    if ignored:
        st.caption("Colonnes ignorées (non numériques ou incomplètes) : " + ", ".join(map(str, ignored)))
    return df, num.to_numpy(dtype=float), list(num.columns)


def standardize(X):
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def project_2d(X):
    """Coordonnées sur les deux premières composantes principales et part de variance expliquée."""
    Xc = X - X.mean(axis=0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    share = (s[:2] ** 2).sum() / (s ** 2).sum()
    return Xc @ vt[:2].T, share
