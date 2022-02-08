import numpy as np

from centroid_summarizer import base
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def certainty_func_average(scores):
    score = 0
    count = 0
    for s in scores:
        if s > 0:
            score += s
            count += 1
    if count > 0:
        score /= count
        return score
    else:
        return 0


def certainty_func_stanford(scores):
    score = 0
    minim = 100000
    for s in scores:
        score += s
        if s < minim & s > 0:
            minim = s
    score /= 1 - minim
    return score


class CentroidWordEmbeddingsSummarizer():
    def __init__(
            self,
            embedding_model,
            length_limit = base.default_length_limit_embeddings,
            topic_threshold = base.default_topic_threshold,
            sim_threshold = base.default_similarity_threshold,
            reordering=True,
            zero_center_embeddings=False,
            keep_first=False,
            bow_param=0,
            length_param=0,
            position_param=0,
            certainty_func = certainty_func_average
    ):
        base.logger.debug("Initializing centroid word embeddings summarizer.")

        self.embedding_model = embedding_model

        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        self.reordering = reordering

        self.keep_first = keep_first
        self.bow_param = bow_param
        self.length_param = length_param
        self.position_param = position_param

        self.certainty_func = certainty_func

        self.zero_center_embeddings = zero_center_embeddings

        if zero_center_embeddings:
            self._zero_center_embedding_coordinates()
        return

    def get_bow(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0
        return tfidf, centroid_vector

    def get_topic_idf(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        base.logger.debug("Max centroid vector: {}".format(centroid_vector.max()))

        feature_names = vectorizer.get_feature_names_out()

        relevant_vector_indices = np.where(centroid_vector > self.topic_threshold)[0]

        word_list = list(np.array(feature_names)[relevant_vector_indices])
        return word_list


    def word_vectors_cache(self, sentences):
        self.word_vectors = dict()
        for s in sentences:
            words = word_tokenize(s)
            for w in words:
                # import ipdb
                # ipdb.set_trace()
                if w in self.embedding_model.wv.key_to_index:
                    if self.zero_center_embeddings:
                        self.word_vectors[w] = (
                            self.embedding_model.wv[w] - self.centroid_space
                        )
                    else:
                        self.word_vectors[w] = self.embedding_model.wv[w]


    # Sentence representation with sum of word vectors
    # By default we'll use
    def compose_vectors(self, words):
        composed_vector = np.zeros(self.embedding_model.wv.vector_size, dtype="float32")
        count = 0
        for w in words:
            if w in self.embedding_model.wv.key_to_index:
                composed_vector = composed_vector + self.embedding_model.wv[w]
                count += 1
        if count != 0:
            composed_vector = np.divide(composed_vector, count)
        return composed_vector

    def summarize(
            self,
            raw_sentences,
            clean_sentences,
            limit = base.default_length_limit
    ):
        base.logger.debug(
            "ORIGINAL TEXT STATS = {0} chars, {1} words, {2} sentences".format(
                len("".join(raw_sentences)),
                sum( len(word_tokenize(_)) for _ in raw_sentences ),
                len(raw_sentences)
            )
        )
        base.logger.debug("*** RAW SENTENCES ***")
        for i, s in enumerate(raw_sentences):
            base.logger.debug("{}: {}".format(str(i).ljust(6), s))
        base.logger.debug("*** CLEAN SENTENCES ***")
        for i, s in enumerate(clean_sentences):
            base.logger.debug("{}: {}".format(str(i).ljust(6), s))

        centroid_words = self.get_topic_idf(clean_sentences)

        base.logger.debug("*** CENTROID WORDS ***")
        base.logger.debug(
            "{} {}".format(
                str(len(centroid_words)).ljust(6),
                centroid_words
            )
        )

        centroid_vector = self.compose_vectors(centroid_words)

        tfidf, centroid_bow = self.get_bow(clean_sentences)
        max_length = max( len(word_tokenize(_)) for _ in clean_sentences )

        sentences_scores = []
        for i in range(len(clean_sentences)):
            scores = []
            words = word_tokenize(clean_sentences[i])
            sentence_vector = self.compose_vectors(words)

            scores.append(base.similarity(sentence_vector, centroid_vector))
            scores.append(self.bow_param * base.similarity(tfidf[i, :], centroid_bow))
            scores.append(self.length_param * (1 - (len(words) / max_length)))
            scores.append(self.position_param * (1 / (i + 1)))
            score = self.certainty_func(scores)

            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))

            base.logger.debug("{}: {}; {}".format(
                str(i).ljust(6),
                scores,
                score
            ))

        sentence_scores_sort = sorted(
            sentences_scores, key=lambda el: el[2], reverse=True
        )
        base.logger.debug("*** SENTENCE SCORES ***")
        for s in sentence_scores_sort:
            base.logger.debug(
                "{}: {}; {}".format(
                    str(s[0]).ljust(6),
                    str(s[1]),
                    str(s[2])
                )
            )

        count = 0
        sentences_summary = []

        if self.keep_first:
            for s in sentence_scores_sort:
                if s[0] == 0:
                    sentences_summary.append(s)
                    count += len(word_tokenize(s[1]))
                    sentence_scores_sort.remove(s)
                    break

        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = base.similarity(s[3], ps[3])
                base.logger.debug(
                    "{}: {}; {}".format(
                        str(s[0]).ljust(6),
                        ps[0],
                        sim
                    )
                )
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                base.logger.debug(
                    "{}: {}".format(
                        str(s[0]).ljust(6),
                        s[1]
                    )
                )
                sentences_summary.append(s)
                count += len(word_tokenize(s[1]))

        if self.reordering:
            sentences_summary = sorted(
                sentences_summary,
                key = lambda _: _[0],
                reverse = False
            )

        # summary = " ".join([s[1] for s in sentences_summary])
        # base.logger.debug(
        #     "SUMMARY TEXT STATS = {0} chars, {1} words, {2} sentences".format(
        #         len(summary), len(word_tokenize(summary)), len(sentences_summary)
        #     )
        # )
        # base.logger.debug("*** SUMMARY ***")
        # base.logger.debug(summary)
        # return summary

        for s in sentences_summary:
            yield(s[1])


    def _zero_center_embedding_coordinates(self):
        # Create the centroid vector of the whole vector space
        count = 0
        self.centroid_space = np.zeros(
            self.embedding_model.vector_size, dtype="float32"
        )
        self.index2word_set = set(self.embedding_model.wv.index2word)
        for w in self.index2word_set:
            self.centroid_space = self.centroid_space + self.embedding_model.wv[w]
            count += 1
        if count != 0:
            self.centroid_space = np.divide(self.centroid_space, count)
