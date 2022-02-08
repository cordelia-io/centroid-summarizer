import logging

from numpy import count_nonzero, dot
from numpy.linalg import norm

logger = logging.getLogger("centroid_summarizer")


def cosine(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def similarity(v1, v2):
    score = 0.0
    if count_nonzero(v1) != 0 and count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


default_language = "english"
default_length_limit = 100
default_length_limit_embeddings = 10
default_placeholder = "\0" # "###nul###"
default_remove_stopwords = True
default_similarity_threshold = 0.95
default_topic_threshold = 0.3


# class BaseSummarizer:
#     def __init__(
#             self,
#             language=default_language,
#             remove_stopwords=default_remove_stopwords,
#             stopwords=default_stopwords,
#             length_limit=default_length_limit,
#             placeholder=default_placeholder
#     ):
#         self.language = language
#         self.remove_stopwords = remove_stopwords
#         self.stopwords = stopwords
#         self.length_limit = length_limit
#         self.placeholder = placeholder
#         if remove_stopwords:
#             stopword_remover = KeywordProcessor()
#             for stopword in stopwords:
#                 stopword_remover.add_keyword(stopword, self.placeholder)
#             self.stopword_remover = stopword_remover
#         return
# 
#     def summarize(self, text, limit=default_length_limit):
#         raise NotImplementedError()
