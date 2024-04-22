import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    confusion_matrix, make_scorer
import seaborn as sns

def process_image(image_path):
    """Traitement d'une image avec SIFT et affichage des résultats."""
    # Chargement de l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None  # Gère le cas où l'image n'est pas chargée correctement

    # Normalisation des valeurs de pixels à l'échelle [0, 1]
    image = image / 255.0

    # Égalisation de l'histogramme pour améliorer le contraste
    image = cv2.equalizeHist((image * 255).astype(np.uint8))

    # Filtrage du bruit avec un filtre bilatéral
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Configuration de SIFT
    sift = cv2.SIFT_create()

    # Détection des points clés et calcul des descripteurs avec SIFT
    kp, des = sift.detectAndCompute(image, None)

    return kp, des, image

def display_keypoints(image, keypoints, title="Image avec Points Clés SIFT"):
    """Affichage des points clés sur l'image."""
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img_with_keypoints, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def process_images_concurrently(image_paths, max_workers=6):
    """Traite les images en parallèle."""
    # Création d'un pool d'exécuteurs avec un nombre maximal de travailleurs
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_image, image_paths))

    sift_descriptors = []

    for keypoints, descriptors, image in results:
        if descriptors is not None:
            sift_descriptors.append(descriptors)

    # Utilisation d'une liste pour gérer les tableaux de tailles différentes
    sift_keypoints_by_img = np.array(sift_descriptors, dtype=object)
    sift_keypoints_all = np.vstack(sift_descriptors) if sift_descriptors else np.array([])

    return sift_keypoints_by_img, sift_keypoints_all

def plot_tsne_grid(data, categories_encoded, label_names, n_rows=2, n_cols=2):
    perplexities = [20, 40, 70, 100]
    # Création d'une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(23, 20))

    # Aplatir la liste des axes pour une indexation facile
    axes = axes.flatten()

    # Mapper les catégories encodées vers les noms réels
    categories_color_mapping = dict(zip(np.unique(categories_encoded), label_names))
    unique_categories = np.unique(categories_encoded)
    category_colors = sns.color_palette('tab10', len(unique_categories))

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label_names[i],
                                  markerfacecolor=category_colors[i], markersize=12)
                       for i in range(len(unique_categories))]

    # Boucle sur chaque valeur de perplexité
    for i, perplexity in enumerate(perplexities):
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(data)
        categories_mapped = np.vectorize(categories_color_mapping.get)(categories_encoded)

        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=categories_mapped,
                        palette=category_colors, ax=axes[i], legend=False, s=50, alpha=0.6)

        axes[i].set_title(f't-SNE avec perplexité = {perplexity}')
        axes[i].set_xlabel('Composante t-SNE 1')
        axes[i].set_ylabel('Composante t-SNE 2')

    # Ajouter la légende en haut à droite de la figure entière, à l'extérieur du dernier sous-graphique
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), title="Catégories")
    plt.tight_layout()
    plt.show()

def perform_kmeans(X_data, true_labels, label_names, n_clusters=7, random_state=42):
    """
    Effectue le clustering K-Means sur des données réduites et évalue les résultats avec le score de silhouette
    et l'ARI.

    Parameters:
    - X_data : les données sur lesquelles appliquer le K-Means.
    - true_labels : les vraies étiquettes de catégories pour le calcul de l'ARI.
    - label_names : les noms réels des catégories pour l'affichage.
    - n_clusters : le nombre de clusters à former.
    - random_state : la graine aléatoire pour la reproductibilité des résultats.

    Returns:
    - silhouette_avg : le score de silhouette moyen pour l'évaluation des clusters.
    - ari_score : l'Adjusted Rand Index score pour l'évaluation des clusters.
    - accuracy : l'accuracy score pour évaluer la précision des clusters alignés avec les vraies catégories.
    """
    # Initialisation de K-Means
    # kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=400, tol=1e-5, algorithm='elkan', init='k-means++',
                    random_state=random_state)

    # Application de K-Means sur les données réduites
    kmeans.fit(X_data)

    # Prédiction des clusters
    clusters = kmeans.predict(X_data)

    def conf_mat_transform(y_true, y_pred):
        conf_mat = metrics.confusion_matrix(y_true, y_pred)
        corresp = np.argmax(conf_mat, axis=0)
        return pd.Series(y_pred).apply(lambda x: corresp[x])

    clusters_aligned = conf_mat_transform(true_labels, clusters)

    silhouette_avg = silhouette_score(X_data, clusters_aligned)
    ari_score = adjusted_rand_score(true_labels, clusters_aligned)
    accuracy = accuracy_score(true_labels, clusters_aligned)

    # Affichage des scores
    print(f'Silhouette Score: {silhouette_avg:.4f}')
    print(f'Adjusted Rand Score: {ari_score:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    # Création de la matrice de confusion
    conf_matrix = confusion_matrix(true_labels, clusters_aligned)
    plt.figure(figsize=(10, 10))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    # Ajout de la visualisation t-SNE colorée par clusters
    X_data = pd.DataFrame(X_data, columns=['tsne1', 'tsne2'])
    X_data['cluster'] = clusters_aligned
    n_clusters = len(np.unique(clusters_aligned))
    palette = sns.color_palette('tab10', n_colors=n_clusters)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="cluster",
        palette=palette,
        s=50, alpha=0.6,
        data=X_data
    )
    plt.title('TSNE selon les clusters', fontsize=30, pad=35, fontweight='bold')
    plt.xlabel('tsne1')
    plt.ylabel('tsne2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 12})
    plt.tight_layout()
    plt.show()

    # Retourne tous les résultats dans un seul dictionnaire
    return {
        'Silhouette Score': silhouette_avg,
        'Adjusted Rand Score': ari_score,
        'Accuracy': accuracy
    }
