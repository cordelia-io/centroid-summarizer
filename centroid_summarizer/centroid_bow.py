import numpy as np

from centroid_summarizer import base
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CentroidBOWSummarizer():

    def __init__(
            self,
            language=base.default_language,
            topic_threshold=base.default_topic_threshold,
            similarity_threshold=base.default_similarity_threshold
    ):

        base.logger.debug("Initializing centroid bag-of-words summarizer.")

        self.topic_threshold = topic_threshold
        self.similarity_threshold = similarity_threshold

    def summarize(self, raw_sentences, clean_sentences, word_count=base.default_word_count):
        raw_sentences = [
            " ".join(_) if type(_) == list else _
            for _ in raw_sentences
        ]
        clean_sentences = [
            " ".join(_) if type(_) == list else _
            for _ in clean_sentences
        ]
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(clean_sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0

        sentences_scores = []
        for i in range(tfidf.shape[0]):
            score = base.similarity(tfidf[i, :], centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :]))

        sentence_scores_sort = sorted(
            sentences_scores,
            key = lambda _: _[2],
            reverse=True
        )

        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if count > word_count:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = base.similarity(s[3], ps[3])
                base.logger.debug(
                    "{}: {}; {}".format(
                        str(s[0]).rjust(6),
                        ps[0],
                        sim
                    )
                )
                if sim > self.similarity_threshold:
                    include_flag = False
                    break
            if include_flag:
                base.logger.debug(
                    "{}: {}".format(
                        str(s[0]).rjust(6),
                        s[1]
                    )
                )
                # sentences_summary.append(s)
                count += len(word_tokenize(s[1]))
                yield s[1]

        # summary = [ s[1] for s in sentences_summary ]
        # return summary
