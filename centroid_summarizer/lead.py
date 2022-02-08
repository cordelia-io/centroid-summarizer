from nltk.tokenize import sent_tokenize, word_tokenize

from centroid_summarizer import base


default_base_length_limit = 10


class LeadSummarizer():
    def __init__(
            self,
            language = base.default_language,
            length_limit = default_base_length_limit,
            remove_stopwords = base.default_remove_stopwords
    ):
        super().__init__(language, remove_stopwords, length_limit)

    def summarize(self, text, limit=default_base_length_limit):
        raw_sentences = sent_tokenize(text)
        count = 0
        # sentences_summary = []
        for s in raw_sentences:
            if count > limit:
                break
            count += len(word_tokenize(s))
            # sentences_summary.append(s)
            yield word_tokenize(s)

        # return " ".join(sentences_summary)
