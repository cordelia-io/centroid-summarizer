import logging

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from numpy import count_nonzero, dot
from numpy.linalg import norm
from unidecode import unidecode

logger = logging.getLogger("centroid_summarizer")


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
    if type(text) == str:
        arr = sent_tokenize(text)
    if not hasattr(text,"__iter__"):
        raise Exception("Expects a string or an array of strings.")
    for sentence in arr:
        if type(sentence) != str:
            sentence = " ".join(sentence)
        clean_sentence = []
        for word in word_tokenize(sentence):
            clean_word = "".join(
                letter for letter in unidecode(
                    str(word).lower()
                )
                if letter.isalpha()
            )
            if clean_word and clean_word not in stopwords:
                clean_sentence.append(clean_word)
        yield " ".join(clean_sentence)


default_language = "english"
default_length_limit = 100
default_length_limit_embeddings = 10
default_placeholder = "\0" # "###nul###"
default_remove_stopwords = True
default_similarity_threshold = 0.95
default_topic_threshold = 0.3
