import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import VGG16
from keras.optimizers import RMSprop



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


def plot_tsne(data, categories_encoded, label_names, perplexity):
    # Création de la figure
    plt.figure(figsize=(6, 6))

    # Mapper les catégories encodées vers les noms réels
    categories_color_mapping = dict(zip(np.unique(categories_encoded), label_names))
    unique_categories = np.unique(categories_encoded)
    category_colors = sns.color_palette('tab10', len(unique_categories))

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=label_names[i],
                                  markerfacecolor=category_colors[i], markersize=12)
                       for i in range(len(unique_categories))]

    # Création de l'objet t-SNE avec la perplexité spécifiée
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(data)
    categories_mapped = np.vectorize(categories_color_mapping.get)(categories_encoded)

    # Tracé du scatter plot avec t-SNE
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=categories_mapped,
                    palette=category_colors, legend=False, s=50, alpha=0.6)

    # Ajout de titres et de labels d'axes
    plt.title(f't-SNE avec perplexité = {perplexity}')
    plt.xlabel('Composante t-SNE 1')
    plt.ylabel('Composante t-SNE 2')

    # Ajout de la légende en haut à droite de la figure
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), title="Catégories")

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
    plt.figure(figsize=(8, 6))
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
    plt.figure(figsize=(8, 6))
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

def image_prep_fct(image_paths, preprocess_function, target_size=(224, 224)):
    prepared_images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        img = preprocess_function(img)  # Appliquez la fonction de prétraitement spécifique au modèle
        prepared_images.append(img)
    return np.array(prepared_images)

def prepare_data(paths_train, paths_val, paths_test, preprocess_function, target_size):
    X_train = image_prep_fct(paths_train, preprocess_function, target_size=target_size)
    X_val = image_prep_fct(paths_val, preprocess_function, target_size=target_size)
    X_test = image_prep_fct(paths_test, preprocess_function, target_size=target_size)
    return X_train, X_val, X_test

def train_model(model, X_train, y_train, X_val, y_val, model_save_path):
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, es]
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=callbacks_list, validation_data=(X_val, y_val), verbose=1)
    return model, history

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, best_weights_path):
    # Score du dernier epoch
    loss, accuracy = model.evaluate(X_train, y_train, verbose=True)
    print("Training Accuracy after last epoch: {:.4f}".format(accuracy))
    print()

    # Évaluation sur l'ensemble de test avec les poids finaux après toutes les époques
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    print("Test Accuracy after last epoch: {:.4f}".format(accuracy))
    print()

    # Chargement des poids de l'epoch optimal
    model.load_weights(best_weights_path)

    # Réévaluation sur l'ensemble de validation avec les poids de l'epoch optimal
    loss, accuracy = model.evaluate(X_val, y_val, verbose=True)
    print("Validation Accuracy (best epoch): {:.4f}".format(accuracy))

    # Réévaluation sur l'ensemble de test avec les poids de l'epoch optimal
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    print("Test Accuracy (best epoch): {:.4f}".format(accuracy))

    # Prédire les étiquettes pour l'ensemble de test
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculer l'ARI
    ari_score = adjusted_rand_score(y_test, predicted_labels)
    print("Adjusted Rand Index (ARI): {:.4f}".format(ari_score))

    return loss, accuracy, ari_score

