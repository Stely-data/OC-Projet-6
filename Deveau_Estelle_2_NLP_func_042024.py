import re
import string
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import accuracy_score, confusion_matrix
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Matrice de confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

    # Retourne tous les résultats dans un seul dictionnaire
    return {
        'Silhouette Score': silhouette_avg,
        'Adjusted Rand Score': ari_score,
        'Accuracy': accuracy
    }

def plot_tsne_grid(data, categories_encoded, n_rows=3, n_cols=2):
    perplexities = [20, 30, 40, 60, 70, 100]
    # Création d'une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 30))

    # Aplatir la liste des axes pour une indexation facile
    axes = axes.flatten()

    # Boucle sur chaque valeur de perplexité
    for i, perplexity in enumerate(perplexities):
        # Création de l'objet TSNE avec la perplexité actuelle
        tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(data)

        # Visualisation avec t-SNE sur le sous-graphique correspondant
        scatter = axes[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=categories_encoded, cmap='tab10')

        # Ajout d'un titre au sous-graphique
        axes[i].set_title(f't-SNE avec perplexité = {perplexity}')

        # Ajout des labels des axes
        axes[i].set_xlabel('Composante t-SNE 1')
        axes[i].set_ylabel('Composante t-SNE 2')

        # Création de la légende, ajoutée une seule fois
        if i == 0:
            handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
            fig.legend(handles, labels, loc='upper right', title="Catégories")

    # Ajustement des sous-graphiques pour éviter les chevauchements
    plt.tight_layout()
    plt.show()
