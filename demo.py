#!/usr/bin/env python

import centroid_summarizer
from unidecode import unidecode
from nltk.tokenize import word_tokenize, sent_tokenize


text = "Here are some raw sentences. They should contain all text from a single document. There should be a lot of them! These have not been pre-processed. Blah blah blah."

raw = sent_tokenize(text)

clean = list(centroid_summarizer.simple_clean(text))


cbs = centroid_summarizer.CentroidBOWSummarizer()

print(list(cbs.summarize(
    raw,
    clean,
    limit=20
)))


from gensim.models import Word2Vec

model = Word2Vec(min_count=1)
model.build_vocab(raw)
model.train([ word_tokenize(_) for _ in clean ], total_examples=model.corpus_count, epochs=model.epochs)

cws = centroid_summarizer.CentroidWordEmbeddingsSummarizer(model)

print(list(cws.summarize(
    raw,
    clean,
    limit=20
)))
