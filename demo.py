#!/usr/bin/env python

import centroid_summarizer
from nltk.tokenize import word_tokenize, sent_tokenize

# Courtesy officeipsum.com
text = "Just do what you think. I trust you. The hair is just too polarising. Low resolution? It looks ok on my screen. This turned out different than I decscribed. Will royalties in the company do instead of cash? Appeal to the client. Sue the vice president, is there a way we can make the page feel more introductory without being cheesy? So can my website be in English? Try a more powerful colour. Concept is bang on, but can we look at a better execution? I really like the colour but can you change it, can we try some other colours maybe? I have an awesome idea for a startup, and I need you to build it for me."

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
