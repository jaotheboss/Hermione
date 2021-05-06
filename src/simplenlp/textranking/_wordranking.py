from collections import OrderedDict
from os import stat                     # for data storage
import pandas as pd
import numpy as np                                      # array manipulation
import networkx as nx                                   # matrix manipulation
import spacy                                            # main nlp module
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_sm')


class WordRankerUtil():
    """
    Main class that ranks each keyword of a sentence based on Google's
    Page Ranking algorithm

    It can take in a single sentence to an entire paragraph.
    """
    damping_coef = 0.85                    # recommended
    min_diff = 1e-5                        # convergence threshold
    steps = 64                             # number of iterations
    candidate_pos = ['NOUN', 'PROPN']      # which kinds of words to keep for analysis

    @staticmethod
    def add_candidate_pos(candidate_pos: list) -> None:
        for pos in candidate_pos:
            if pos not in WordRankerUtil.candidate_pos:
                WordRankerUtil.candidate_pos.append(pos)

    @staticmethod
    def set_stopwords() -> None:
        """
        Function:     Set stopwords for the nlp module

        Input:        None

        Returns:      Confirmation (string)
        """
        for word in STOP_WORDS:
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    @staticmethod
    def add_stopwords(stopwords: list) -> None:
        """
        Function:     Add stopwords to the additional list

        Input:        An iterable or a string of stopword(s)

        Return:       Confirmation (string)
        """
        stopwords = set(stopwords)
        for new in stopwords:
            lexeme = nlp.vocab[new]
            lexeme.is_stop = True

        # add everything to the stop_words set
        STOP_WORDS = STOP_WORDS.union(stopwords)

    @staticmethod
    def segment_sentences(nlp_doc: object) -> list:
        """
        Function:     Filter each sentence into their essence.
                      Meaning non stop words and words that are
                      useful.

        Inputs:       Spacy NLP doc object

        Returns:      A list of sentence keywords in list form
                      eg. [[keywords in sentence], [keywords in sentence], ...]
        """
        sentence_key_words = []
        for sentence in nlp_doc.sents:
            key_words = []
            for token in sentence:
                # will only store non stopwords and those with candidate POS tag
                if token.pos_ in WordRankerUtil.candidate_pos and token.is_stop is False:
                    key_words.append(token.text.lower())
            sentence_key_words.append(key_words)
        return sentence_key_words

    @staticmethod
    def get_vocabulary(segmented_sentences: list) -> OrderedDict:
        """
        Function:     Get the vocabulary of all sentences

        Input:        A list of sentences

        Return:       Ordered dictionary of the vocabulary
        """
        vocab = OrderedDict()
        index = 0
        for sentence in segmented_sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = index  # labelling word with index value
                    index += 1
        return vocab

    @staticmethod
    def build_token_pairs(segmented_sentences: list, window_size: int) -> list:
        """
        Function:     To pair up the keywords; to create a
                      network of these keywords so as to
                      execute Google's page ranking algo

        Inputs:       List of sentence keywords in a list (List)
                      window_size; how many words at a time to
                      look at to pair (int)

        Returns:      A list of token pairs; keyword pairs
        """
        token_pairs = []
        for sentence in segmented_sentences:
            for i, keyword in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (keyword, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    @staticmethod
    def build_keyword_matrix(vocab: OrderedDict, token_pairs: list) -> np.ndarray:
        """
        Function:     Create a matrix to show the relationship
                      between the keywords.

                      Based on the vocabulary and window size

        Inputs:       Indexed vocabulary and token pairs

        Returns:      Matrix of token pairs
        """
        # Building the matrix
        vocab_size = len(vocab)
        matrix = np.zeros(
            (vocab_size, vocab_size),
            dtype='float'
        )
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            matrix[i][j] = 1

        # Symmetrizing the matrix
        matrix = matrix + matrix.T - np.diag(matrix.diagonal())

        # Normalizing the columns of the matrix
        norm = np.sum(matrix, axis=0)
        matrix = np.divide(
            matrix,
            norm,
            where=norm != 0
        )
        return matrix

    def run_text_rank(text: str, window_size: int, use_nx: bool = False) -> dict:
        """
        Function:     Combines all the methods to create the
                      matrix out of the token pairs and
                      performs the Page Ranking algo

        Input:        The text that is to be analyzed (string)
                      A window_size (int)

        Returns:      None. Sets the node_weight attribute of
                      the class
        """
        # set stopwords
        WordRankerUtil.set_stopwords()

        # parse text with spaCy and create doc object
        doc = nlp(text)

        # segment the sentences for keywords
        sentences = WordRankerUtil.segment_sentences(doc)

        # build the dictionary out of the keywords only
        vocab = WordRankerUtil.get_vocabulary(sentences)

        # get the token pairs from the windows out of the keywords only
        token_pairs = WordRankerUtil.build_token_pairs(sentences, window_size)

        # initialising the matrix and the pagerank values
        matrix = WordRankerUtil.build_keyword_matrix(vocab, token_pairs)

        # iteration of the weights via the pageranking algorithm
        if use_nx:
            pagerank_values = nx.pagerank(nx.from_numpy_array(matrix))
        else:
            pagerank_values = np.array([1] * len(vocab))
            previous_pagerank_sum = 0
            for epoch in range(WordRankerUtil.steps):
                # execute pageranking algorithm
                pagerank_values = (
                    1 - WordRankerUtil.damping_coef) + WordRankerUtil.damping_coef * np.dot(matrix, pagerank_values)
                current_pagerank_sum = sum(pagerank_values)
                if abs(previous_pagerank_sum - current_pagerank_sum) < WordRankerUtil.min_diff:
                    break
                else:
                    previous_pagerank_sum = current_pagerank_sum

        # extract weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pagerank_values[index]

        return node_weight


# Ranking Function
def rank_words(text: str,
               n_words: int = 10,
               window_size: int = 5,
               use_nx: bool = False) -> dict:
    """
    A streamlined function for simple calls.

    Calls on the WordRanker constructor to run pagerank on the text. 
    """
    word_weight_matrix = WordRankerUtil.run_text_rank(text, window_size, use_nx)
    word_weight_matrix = OrderedDict(
        sorted(
            word_weight_matrix.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )
    result = {}
    for i, (key, value) in enumerate(word_weight_matrix.items()):
        if pd.isna(value):
            value = 0
        result[i] = {'word': key, 'score': int(value*1000)/1000}
        if i + 1 >= n_words:
            break
    return result
