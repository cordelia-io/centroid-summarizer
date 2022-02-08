import logging

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from numpy import count_nonzero, dot
from numpy.linalg import norm
from unidecode import unidecode

logging.basicConfig()
logger = logging.getLogger("centroid_summarizer")

# from scipy.spatial.distance import cosine
def cosine(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def similarity(v1, v2):
    score = 0.0
    if count_nonzero(v1) != 0 and count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score

def simple_clean(
        text,
        stopwords=nltk_stopwords.words("english")
):
    err = Exception("Expects a string, or an array of strings, or an array of arrays of strings.")
    if type(text) == str:
        arr = sent_tokenize(text)
    else:
        arr = text
    if not hasattr(arr, "__iter__"):
        raise err
    for sentence in arr:
        if type(sentence) != str:
            try:
                sentence = " ".join(sentence)
            except:
                raise err
        clean_sentence = []
        for word in word_tokenize(sentence):
            clean = " ".join(
                "".join(
                    letter
                    if letter.isalpha()
                    else " "
                    for letter in unidecode(
                            str(word).lower()
                    )
                ).split()
            )
            if clean:
                if " " in clean:
                    for _ in clean.split():
                        if _ not in stopwords:
                            clean_sentence.append(_)
                else:
                    if clean not in stopwords:
                        clean_sentence.append(clean)
        yield clean_sentence


default_language = "english"
default_word_count = 150
default_similarity_threshold = 0.95
default_topic_threshold = 0.3
