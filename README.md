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

# Uncomment this if you want to see a ton of debugging messages:
# centroid_summarizer.logger.setLevel(10)


# Text from the proof-of-concept code at
# https://gist.github.com/vinimonteiro/3898ce27023ec4241c4879dac67ca27d
doc = """In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning. According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills."""

raw_sentences = sent_tokenize(doc)

clean = list(centroid_summarizer.simple_clean(raw_sentences))

cbs = centroid_summarizer.CentroidBOWSummarizer()

print("\nBag-of-words summary:\n")
print(" ".join(
    list(cbs.summarize(
        raw_sentences,
        [ " ".join(_) for _ in clean ]
    ))
),end="\n\n")


from gensim.models import Word2Vec
model = Word2Vec(clean, min_count=1)

cws = centroid_summarizer.CentroidWordEmbeddingsSummarizer(
    model
)

print("Embedding summary:\n")
print(" ".join(
    list(cws.summarize(
        raw_sentences,
        [ " ".join(_) for _ in clean ]
    ))
),end="\n\n")

```

Output from the above:

```
Bag-of-words summary:

This will require more collaborations and training and working with AI. That’s
why it has become more critical than ever for educational institutions to 
integrate new cloud and AI technologies. According to Manish Prakash, Country 
General Manager-PS, Health and Education, Microsoft India, said, "With AI being
the defining technology of our time, it is transforming lives and industry and
the jobs of tomorrow will require a different skillset. The program was 
developed to provide job ready skills to programmers who wanted to hone their
skills in AI and data science with a series of online courses which featured
hands-on labs and expert instructors as well. The program is an attempt to ramp
up the institutional set-up and build capabilities among the educators to
educate the workforce of tomorrow." This program also included 
developer-focused AI school that provided a bunch of assets to help build AI
skills.

Embedding summary:

In an attempt to build an AI-ready workforce, Microsoft announced Intelligent
Cloud Hub which has been launched to empower the next generation of students
with AI-ready skills. Envisioned as a three-year collaborative program,
Intelligent Cloud Hub will support around 100 institutions with AI 
infrastructure, course content and curriculum, developer support, development
tools and give students access to cloud and AI services. This will require more
collaborations and training and working with AI. That’s why it has become more
critical than ever for educational institutions to integrate new cloud and AI
technologies.
```


## About

This package is derived from the [original implementation](https://github.com/gaetangate/text-summarizer) by the authors of the paper "Centroid-based Text Summarization through Compositionality of Word Embeddings" accepted at MultiLing Workshop at EACL 2017. http://www.aclweb.org/anthology/W17-1003

The method is described in [A Better Approach to Text Summarization](https://towardsdatascience.com/a-better-approach-to-text-summarization-d7139b571439) by [@vinimonteiro](https://github.com/vinimonteiro).


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
