import re
import string
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

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
