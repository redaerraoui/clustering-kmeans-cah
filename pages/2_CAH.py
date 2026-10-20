import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score

from modules.data_utils import load_numeric, standardize
from modules.export_utils import export_pdf, export_zip, figure_png

st.title("🌳 Classification Hiérarchique (CAH)")

uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])
c1, c2 = st.columns(2)
linkage_method = c1.selectbox("Méthode de liaison", ["ward", "single", "complete", "average"])
k = c2.slider("Nombre de clusters (k)", 2, 10, 3)
scale = st.checkbox("Centrer et réduire les variables", value=True)

if uploaded_file:
    loaded = load_numeric(uploaded_file)
    if loaded:
        df, X_raw, cols = loaded
        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        X = standardize(X_raw) if scale else X_raw
        Z = linkage(X, method=linkage_method)

        # Ligne de coupe entre la fusion qui laisse k groupes et celle qui en laisse k - 1
        cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2
        st.subheader("Dendrogramme")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, ax=ax, color_threshold=cut, no_labels=X.shape[0] > 60)
        ax.axhline(y=cut, color="red", linestyle="--")
        ax.set_ylabel("Distance de fusion")
        st.pyplot(fig)

        labels = fcluster(Z, k, criterion="maxclust")
        df_clusters = df.copy()
        df_clusters["cluster"] = labels
        st.subheader("📋 Affectation des observations")
        st.dataframe(df_clusters.head())

        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else float("nan")
        log_lines = [
            f"Données : {X.shape[0]} lignes, {X.shape[1]} variables numériques ({', '.join(map(str, cols))}).",
            f"Variables centrées-réduites : {'oui' if scale else 'non'}",
            f"CAH avec méthode de liaison : {linkage_method}",
            f"Nombre de clusters demandés : {k} (hauteur de coupe {cut:.3f})",
            f"Silhouette moyenne : {sil:.3f}",
            f"Effectifs : {[int((labels == i).sum()) for i in range(1, k + 1)]}",
        ]
        st.text("\n".join(log_lines))

        png = figure_png(fig)
        col1, col2 = st.columns(2)
        col1.download_button("📤 Télécharger le PDF", export_pdf("Résultat de la CAH", log_lines, png),
                             file_name="result_cah.pdf", mime="application/pdf")
        col2.download_button("📦 Télécharger le ZIP", export_zip("\n".join(log_lines), df_clusters, png, "cah"),
                             file_name="result_cah.zip", mime="application/zip")
else:
    st.info("Importez un fichier CSV ou Excel pour commencer (exemple : data/iris.csv).")
