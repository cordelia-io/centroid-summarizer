#!/usr/bin/env python

import centroid_summarizer
from unidecode import unidecode
from nltk.tokenize import word_tokenize, sent_tokenize


cbs = centroid_summarizer.CentroidBOWSummarizer()


stopwords = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by', 'do', 'does', 'doing', 'for', 'from', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'or', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'you', 'your', 'yours', 'yourself', 'yourselves']

def simple_clean(arr):
    ret = []
    for sentence in arr:
        clean_sentence = []
        for word in sentence:
            w = unidecode(str(word).lower())
            clean_word = "".join(
                letter for letter in w
                if letter.isalpha()
            )
            if len(clean_word) > 3 and clean_word not in stopwords:
                clean_sentence.append(clean_word)
        ret.append(clean_sentence)
    return ret


text = "Here are some raw sentences. They should contain all text from a single document. There should be a lot of them! These have not been pre-processed. Blah blah blah."

raw = sent_tokenize(text)

clean = simple_clean([ word_tokenize(_) for _ in raw ])


print(list(cbs.summarize(
    raw,
    [ " ".join(_) for _ in clean ],
    limit=20
)))


from gensim.models import Word2Vec

model = Word2Vec(min_count=1)
model.build_vocab(raw)
model.train(clean, total_examples=model.corpus_count, epochs=model.epochs)

cws = centroid_summarizer.CentroidWordEmbeddingsSummarizer(model)

print(list(cws.summarize(
    raw,
    [ " ".join(_) for _ in clean ],
    limit=20
)))
