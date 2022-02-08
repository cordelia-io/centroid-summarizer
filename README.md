# Extractive text summarization using centroid distance

## Install


```sh
python3 -m venv venv
source venv/bin/activate
pip install -U pip # optional but recommended
pip install centroid_summarizer
```

## Usage

```python
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

```


## About

This package is derived from the original implementation by the authors of the paper "Centroid-based Text Summarization through Compositionality of Word Embeddings" accepted at MultiLing Workshop at EACL 2017. http://www.aclweb.org/anthology/W17-1003

Original author: Gaetano Rossiello [gaetano.rossiello@ibm.com](mailto:gaetano.rossiello@ibm.com)

## Tutorial

The method is described in this step-by-step guide: [A Better Approach to Text Summarization](https://towardsdatascience.com/a-better-approach-to-text-summarization-d7139b571439)


## Citation


```bibtex
@inproceedings{DBLP:conf/acl-multiling/RossielloBS17,
  author    = {Gaetano Rossiello and
               Pierpaolo Basile and
               Giovanni Semeraro},
  title     = {Centroid-based Text Summarization through Compositionality of Word
               Embeddings},
  booktitle = {MultiLing at EACL},
  pages     = {12--21},
  publisher = {Association for Computational Linguistics},
  year      = {2017}
}
```
