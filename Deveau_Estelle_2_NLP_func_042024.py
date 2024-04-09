import re
import string
import nltk
from nltk.tokenize import word_tokenize, WordPunctTokenizer, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

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
