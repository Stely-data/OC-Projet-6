# traitement d'image
import cv2
from keras.preprocessing.image import load_img, img_to_array

# traitement de données
import numpy as np
import pandas as pd

# ML
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from scikeras.wrappers import KerasClassifier

# deep learning
from keras.applications import VGG16, InceptionResNetV2, DenseNet201
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

# visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# optimisation
from scipy.optimize import linear_sum_assignment

# temps et ressources processus
import time
from concurrent.futures import ThreadPoolExecutor


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
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=2000, init='random')
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

    return tsne_results


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
    kmeans = KMeans(n_clusters=n_clusters, n_init=100, random_state=random_state)

    # Application de K-Means sur les données réduites
    kmeans.fit(X_data)

    # Prédiction des clusters
    clusters = kmeans.predict(X_data)

    def conf_mat_transform(y_true, y_pred):
        # Calcul de la matrice de confusion
        conf_mat = confusion_matrix(y_true, y_pred)

        # Note : linear_sum_assignment minimise le coût, donc nous utilisons -conf_mat pour maximiser les correspondances
        row_ind, col_ind = linear_sum_assignment(-conf_mat)

        # row_ind[i] correspond à la vraie catégorie, col_ind[i] à la catégorie prédite
        mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}

        # Transformer y_pred en utilisant le mapping trouvé
        y_pred_transformed = pd.Series(y_pred).apply(lambda x: mapping.get(x, x))

        return y_pred_transformed

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
        img = preprocess_function(img)
        prepared_images.append(img)
    return np.array(prepared_images)


def prepare_data(paths_train, paths_val, paths_test, preprocess_function, target_size):
    X_train = image_prep_fct(paths_train, preprocess_function, target_size=target_size)
    X_val = image_prep_fct(paths_val, preprocess_function, target_size=target_size)
    X_test = image_prep_fct(paths_test, preprocess_function, target_size=target_size)
    return X_train, X_val, X_test


def create_model_fct(base_model_name='VGG16'):
    """
    Crée et compile un modèle de classification d'images basé sur un modèle CNN pré-entraîné.

    Args:
    - base_model_name (str): Nom du modèle CNN pré-entraîné à utiliser.

    Returns:
    - model: Le modèle Keras compilé.
    """

    # Sélection du modèle de base en fonction du nom fourni
    if base_model_name == 'InceptionResNetV2':
        base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(299, 299, 3))
    elif base_model_name == 'DenseNet201':
        base_model = DenseNet201(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    elif base_model_name == 'EfficientNetB7':
        base_model = EfficientNetB7(include_top=False, weights="imagenet", input_shape=(600, 600, 3))
    else:  # Le modèle par défaut est VGG16 si aucun nom valide n'est fourni
        base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    # extraction des features
    for layer in base_model.layers:
        layer.trainable = False

    # Construction du modèle
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(7, activation='softmax')(x)

    # Définition du nouveau modèle
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compilation du modèle
    model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy"])

    # Affichage du résumé du modèle
    # model.summary()

    return model


def train_model(model, X_train, y_train, X_val, y_val, model_save_path):
    # Début du chronométrage
    start_time = time.time()
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    callbacks_list = [checkpoint, es]
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, callbacks=callbacks_list, validation_data=(X_val, y_val), verbose=1)

    # Fin du chronométrage
    end_time = time.time()

    # Calcul de la durée
    duration = end_time - start_time

    return model, history, duration


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, best_weights_path='none'):
    if best_weights_path != 'none':
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
    print("Validation Accuracy (best): {:.4f}".format(accuracy))

    # Réévaluation sur l'ensemble de test avec les poids de l'epoch optimal
    loss, accuracy = model.evaluate(X_test, y_test, verbose=True)
    print("Test Accuracy (best): {:.4f}".format(accuracy))

    # Prédire les étiquettes pour l'ensemble de test
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculer l'ARI
    true_labels = np.argmax(y_test, axis=1)
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print("Adjusted Rand Index (ARI): {:.4f}".format(ari_score))

    return loss, accuracy, ari_score


def test_hyperparameters(model, X_train, y_train, X_val, y_val, model_save_path):
    learning_rates = [0.01, 0.001]
    batch_sizes = [32, 64]
    epochs_list = [10, 20]

    best_accuracy = 0
    best_duration = float('inf')
    best_params = {}

    for lr in learning_rates:
        optimizer = RMSprop(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                print(f"Testing with learning_rate={lr}, batch_size={batch_size}, epochs={epochs}")

                # Entraînement du modèle
                start_time = time.time()
                history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                                    validation_data=(X_val, y_val), verbose=0)
                duration = time.time() - start_time

                # Évaluation sur les données de validation
                val_accuracy = max(history.history['val_accuracy'])

                # Mise à jour du meilleur modèle si nécessaire
                if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and duration < best_duration):
                    best_accuracy = val_accuracy
                    best_duration = duration
                    best_params = {'learning_rate': lr, 'batch_size': batch_size, 'epochs': epochs}
                    # Sauvegarde du meilleur modèle
                    model.save(model_save_path)

                print(f"Finished {lr}, {batch_size}, {epochs} with val_accuracy={val_accuracy}, duration={duration}")

    print(f"Best parameters: {best_params}")

    # Recharger le meilleur modèle sauvegardé
    best_model = load_model(model_save_path)

    return best_model, best_duration


def plot_model_performance(data_metrics):
    """
    Affiche un graphique à barres des performances des modèles en fonction des différentes métriques,
    en triant et en annotant les résultats pour une meilleure visualisation et comparaison.

    Parameters:
    - data_metrics (pd.DataFrame): DataFrame contenant les colonnes 'Model', 'Metric', et 'Score'.

    """
    # Trier les données pour la visualisation en fonction du score 'Adjusted Rand Score'
    sorted_methods = data_metrics[data_metrics['Metric'] == 'Adjusted Rand Score'] \
        .sort_values(by='Score', ascending=False)['Model'] \
        .unique()

    # Assurer l'ordre des méthodes dans le DataFrame pour le graphique
    data_metrics['Model'] = pd.Categorical(data_metrics['Model'], categories=sorted_methods, ordered=True)
    data_metrics = data_metrics.sort_values('Model')

    # Création du graphique
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(data=data_metrics, x='Metric', y='Score', hue='Model', palette='deep')

    # Ajouter des valeurs de score sur les barres pour une meilleure lisibilité
    for p in barplot.patches:
        if p.get_height() > 0:  # Assurez-vous que la hauteur est positive pour afficher le texte
            barplot.annotate(format(p.get_height(), '.2f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9),
                             textcoords='offset points')

    # Configuration finale du graphique
    plt.title('Comparaison des Métriques de Clustering pour Différentes Méthodes de Traitement d\'Images')
    plt.xlabel('Métrique')
    plt.ylabel('Score')
    plt.legend(title='Modèle', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

