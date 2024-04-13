import re
import string
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, \
    confusion_matrix, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def process_text(text, clean=True, tokenize_method=None, remove_stopwords=False, stemming=False, lemmatization=False,
                 words_to_remove=None):
    """
    Fonction pour prétraiter un texte en appliquant une série d'opérations optionnelles,
    y compris la suppression de mots spécifiques et de mots ne respectant pas une longueur minimale.

    Paramètres :
    - text (str) : Le texte à traiter.
    - clean (bool) : Si True, nettoie le texte en le convertissant en minuscules, retirant ponctuation et chiffres.
    - tokenize_method (str) : Méthode de tokenisation à utiliser ('word_tokenize', 'wordpunct', 'regex').
    - remove_stopwords (bool) : Si True, supprime les stop words du texte tokenisé.
    - stemming (bool) : Si True, applique le stemming aux tokens.
    - lemmatization (bool) : Si True, applique la lemmatisation aux tokens.
    - words_to_remove (list) : Liste optionnelle de mots à supprimer du texte tokenisé.

    Retourne :
    - list : Une liste de tokens après traitement.
    """
    if clean:
        # Nettoyage de base du texte
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)

    if tokenize_method:
        # Tokenisation
        tokens = []
        if tokenize_method == 'word_tokenize':
            tokens = word_tokenize(text)
        elif tokenize_method == 'wordpunct':
            tokenizer = WordPunctTokenizer()
            tokens = tokenizer.tokenize(text)
        elif tokenize_method == 'regex':
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = tokenizer.tokenize(text)

        if remove_stopwords:
            # Suppression des stop words
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]

        if stemming:
            # Application du stemming
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) for token in tokens]

        if lemmatization:
            # Application de la lemmatisation
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        if words_to_remove:
            # Suppression des mots spécifiés
            tokens = [token for token in tokens if token not in words_to_remove]

        return tokens

    return text


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
    - clusters : les étiquettes de clusters prédites par K-Means.
    """
    # Initialisation de K-Means
    # kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans = KMeans(n_clusters=7, n_init=50, max_iter=400, tol=1e-5, algorithm='elkan', init='k-means++',
                    random_state=42)

    # Application de K-Means sur les données réduites
    kmeans.fit(X_data)

    # Prédiction des clusters
    clusters = kmeans.predict(X_data)

    def conf_mat_transform(y_true, y_pred):
        conf_mat = metrics.confusion_matrix(y_true, y_pred)
        corresp = np.argmax(conf_mat, axis=0)

        return pd.Series(y_pred).apply(lambda x: corresp[x])

    # Réaligner les étiquettes de clusters prédites avec les vraies étiquettes
    clusters_aligned = conf_mat_transform(true_labels, clusters)

    # Évaluation avec la mesure Silhouette
    silhouette_avg = silhouette_score(X_data, clusters_aligned)
    print(f'Silhouette Score: {silhouette_avg:.4f}')

    # Comparaison avec les vraies catégories si disponibles
    ari_score = adjusted_rand_score(true_labels, clusters_aligned)
    print(f'Adjusted Rand Score: {ari_score:.4f}')

    # Calcul de l'accuracy
    accuracy = accuracy_score(true_labels, clusters_aligned)
    print(f'Accuracy: {accuracy:.4f}')

    # Création de la matrice de confusion
    conf_matrix = confusion_matrix(true_labels, clusters_aligned)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_names,
                yticklabels=label_names)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    return silhouette_avg, ari_score, accuracy, clusters


def perform_model_with_cross_validation(model, X_train, y_train, X_test, y_test, scoring=None, cv=5):
    """
    Entraîne un modèle sur des données d'entraînement en utilisant la validation croisée et évalue les résultats
    sur un ensemble de test avec plusieurs métriques.

    Parameters:
    - model : le modèle à entraîner.
    - X_train : les features d'entraînement (matrice BoW ou TF-IDF).
    - y_train : les étiquettes d'entraînement.
    - X_test : les features de test.
    - y_test : les étiquettes de test.
    - scoring : Liste de métriques à utiliser pour l'évaluation (par défaut, utilise Accuracy, Adjusted Rand Score,
                Recall, Precision, et F1 Score).
    - cv : Nombre de folds pour la validation croisée (par défaut, 5).

    Returns:
    - metrics : Dictionnaire contenant les métriques calculées.
    """
    if scoring is None:
        # Définition des métriques par défaut
        scoring = {
            'Accuracy': make_scorer(accuracy_score),
            'Adjusted Rand Score': make_scorer(adjusted_rand_score),
            'Recall': make_scorer(recall_score, average='weighted'),
            'Precision': make_scorer(precision_score, average='weighted'),
            'F1 Score': make_scorer(f1_score, average='weighted')
        }

    # Entraînement du modèle avec validation croisée
    cv_results = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, return_train_score=False)

    # Prédictions sur l'ensemble de test
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calcul des métriques sur l'ensemble de test
    accuracy = accuracy_score(y_test, predictions)
    ari_score = adjusted_rand_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Affichage des métriques
    print("Cross-Validation Scores:")
    for metric, values in cv_results.items():
        print(f"{metric}: {np.mean(values):.4f} (± {np.std(values):.4f})")

    print("\nTest Set Metrics:")
    metrics = {
        'Accuracy': accuracy,
        'Adjusted Rand Score': ari_score,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1
    }
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

    # Extraction des noms uniques des catégories
    category_names = np.unique(np.concatenate((y_test, y_train)))

    # Création de la matrice de confusion
    conf_matrix = confusion_matrix(y_test, predictions, labels=category_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=category_names,
                yticklabels=category_names)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    return metrics
