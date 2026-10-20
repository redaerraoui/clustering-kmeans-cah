import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import silhouette_score

from modules.data_utils import load_numeric, project_2d, standardize
from modules.export_utils import export_pdf, export_zip, figure_png
from modules.kmeans_manual import KMeans

st.title("🔶 K-Means Clustering Manuel")

uploaded_file = st.file_uploader("Importer un fichier CSV ou Excel", type=["csv", "xlsx"])
c1, c2, c3 = st.columns(3)
distance = c1.selectbox("Distance utilisée", ["euclidean", "city_block"],
                        help="city_block : distance de Manhattan, centres mis à jour par la médiane (K-médianes).")
k = c2.slider("Nombre de clusters (k)", 2, 10, 3)
seed = c3.number_input("Graine aléatoire", value=42, step=1)
scale = st.checkbox("Centrer et réduire les variables", value=True)

if uploaded_file:
    loaded = load_numeric(uploaded_file)
    if loaded:
        df, X_raw, cols = loaded
        st.subheader("Aperçu des données")
        st.dataframe(df.head())

        X = standardize(X_raw) if scale else X_raw
        model = KMeans(k=k, distance_metric=distance, random_state=int(seed))
        labels, centroids, inertias = model.fit(X)

        # Affichage : les deux variables si d = 2, sinon le plan des deux premières composantes principales
        if X.shape[1] == 2:
            P, C, xl, yl, sub = X, centroids, cols[0], cols[1], ""
        else:
            P, share = project_2d(np.vstack([X, centroids]))
            P, C = P[:-k], P[-k:]
            xl, yl, sub = "Composante 1", "Composante 2", f" (ACP, {share:.0%} de la variance)"

        st.subheader("📊 Résultat du Clustering" + sub)
        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("tab10")
        for i in np.unique(labels):
            ax.scatter(P[labels == i, 0], P[labels == i, 1], s=18, label=f"Cluster {i}", color=cmap(i % 10))
        ax.scatter(C[:, 0], C[:, 1], c="white", edgecolors="black", marker="X", s=120, label="Centres")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title("Résultat du K-Means")
        ax.legend(fontsize=8)
        st.pyplot(fig)

        df_clusters = df.copy()
        df_clusters["cluster"] = labels
        st.subheader("📋 Affectation des observations")
        st.dataframe(df_clusters.head())

        st.subheader("📉 Courbe de convergence")
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, len(inertias) + 1), inertias, marker="o")
        ax2.set_xlabel("Itérations")
        ax2.set_ylabel("Inertie intra-classe" if distance == "euclidean" else "Somme des distances L1")
        st.pyplot(fig2)

        sil = silhouette_score(X, labels, metric="euclidean" if distance == "euclidean" else "cityblock") \
            if len(np.unique(labels)) > 1 else float("nan")
        log_lines = [
            f"Données : {X.shape[0]} lignes, {X.shape[1]} variables numériques ({', '.join(map(str, cols))}).",
            f"Variables centrées-réduites : {'oui' if scale else 'non'}",
            f"K-Means avec distance : {distance}, initialisation K-means++, graine {int(seed)}",
            f"Nombre de clusters : {k}",
            f"Itérations : {model.n_iter_}",
            f"Critère final : {inertias[-1]:.2f}",
            f"Silhouette moyenne : {sil:.3f}",
            f"Effectifs : {[int((labels == i).sum()) for i in range(k)]}",
        ]
        st.text("\n".join(log_lines))

        png = figure_png(fig)
        col1, col2 = st.columns(2)
        col1.download_button("📤 Télécharger le PDF", export_pdf("Résultat du K-Means", log_lines, png),
                             file_name="result_kmeans.pdf", mime="application/pdf")
        col2.download_button("📦 Télécharger le ZIP", export_zip("\n".join(log_lines), df_clusters, png, "kmeans"),
                             file_name="result_kmeans.zip", mime="application/zip")
else:
    st.info("Importez un fichier CSV ou Excel pour commencer (exemple : data/iris.csv).")
