# Clustering K-means et classification ascendante hiérarchique

Projet d'analyse de données réalisé par **Reda Erraoui** en première année de la filière Finance et Ingénierie Décisionnelle de l'ENSA Agadir (2024–2025), sous l'encadrement de Pr. B. El Asri.

**[Essayer le simulateur en ligne](https://redae03.github.io/#simulateur-clustering)** · **[Lire le rapport (PDF)](rapport.pdf)**

## Contenu

Application Streamlit en trois pages :

- **Accueil** (`app.py`) : présentation et navigation.
- **K-Means** (`pages/1_KMeans.py`) : K-means implémenté à la main (`modules/kmeans_manual.py`), avec initialisation K-means++, distance euclidienne ou de Manhattan et courbe de convergence.
- **CAH** (`pages/2_CAH.py`) : classification ascendante hiérarchique avec scipy, au choix entre les liaisons de Ward, simple, complète et moyenne, avec dendrogramme et coupe en k groupes.

Chaque page affiche les affectations, un journal (effectifs, critère, silhouette) et permet d'exporter les résultats en PDF ou en ZIP (journal, CSV des affectations et figure).

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

Importez ensuite un fichier CSV ou Excel. Le jeu `data/iris.csv` (150 fleurs, 4 mesures) peut servir d'exemple.

Sur Iris centré-réduit avec k = 3, on obtient les résultats suivants :

| Méthode | Inertie intra-classe | Silhouette | ARI (accord avec les espèces) |
|---|---:|---:|---:|
| K-means (meilleur de 20 initialisations) | 139,82 | 0,460 | 0,620 |
| CAH, Ward | 148,88 | 0,447 | 0,615 |
| CAH, liaison moyenne | 198,18 | 0,480 | 0,562 |

## Corrections apportées (octobre 2026)

- **Inertie.** L'historique additionnait des distances et non des distances au carré, ce qui ne correspond pas au critère J du rapport. C'est désormais la somme des carrés des distances (ou la somme des distances L1 avec Manhattan). La dernière itération est aussi enregistrée. Auparavant, une convergence dès la première itération laissait l'historique vide et faisait planter `inertias[-1]`.
- **Distance de Manhattan.** Avec cette distance, les centres sont maintenant mis à jour par la médiane (K-médianes), qui minimise la somme des distances L1. La moyenne n'était pas cohérente avec cette distance.
- **Cluster vide.** `X[labels == i].mean()` renvoyait NaN et cassait l'algorithme. Un cluster vide est désormais replacé sur le point le plus éloigné de son centre.
- **Aléatoire.** `np.random.seed` modifiait l'état global de numpy. Un générateur local `np.random.default_rng` le remplace, et la graine est réglable dans l'interface.
- **Données.** `df.values` envoyait aussi les colonnes texte à l'algorithme, par exemple l'espèce dans Iris, ce qui faisait planter le calcul. Seules les colonnes numériques complètes sont maintenant utilisées. Une option permet de centrer et réduire les variables : sans elle, les variables de grande amplitude dominent la distance.
- **Visualisation.** Le nuage affichait seulement les deux premières colonnes. Au-delà de deux variables, il affiche maintenant le plan des deux premières composantes principales, avec la part de variance expliquée. La palette de 5 couleurs, qui se répétait pour k > 5, est remplacée par une palette de 10 couleurs.
- **Dendrogramme.** La ligne de coupe était tracée exactement à la hauteur d'une fusion (`Z[-k, 2]`). Elle est maintenant placée à mi-hauteur entre les deux fusions qui encadrent les k groupes.
- **Export.** Les PDF et ZIP sont générés en mémoire et proposés directement au téléchargement. Avant, un bouton apparaissait dans un autre bouton et le téléchargement disparaissait à la relance de la page. Le PDF utilise `fpdf2`.

## Contact

Reda Erraoui · [Portfolio](https://redae03.github.io) · redaerraouie@gmail.com
