from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import numpy as np                                      # array manipulation
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # similarity computation
import networkx as nx                                   # matrix manipulation
import re                                               # text cleaning
import spacy                                            # main nlp module
nlp = spacy.load('en_core_web_sm')


class SentenceRankerUtil():
    """
    Utility class for sentence ranking
    """
    verbose = False
    embedding_model = SentenceTransformer(
        'paraphrase-distilroberta-base-v1')

    @staticmethod
    def doc_to_sentence(doc: object) -> [list, list]:
        """
        Function:     Extracts the sentences from a nlp doc object and 
                      cleans them before returning

        Inputs:       A nlp doc object from spaCy

        Returns:      Cleaned sentences (string)
        """
        dirty_sentences = [i.text.strip() for i in doc.sents]
        clean_sentences = []
        for dirt in dirty_sentences:
            clean = " ".join([i.lower() for i in dirt])

            # removing anything that's not a word
            clean = re.sub("[^a-zA-Z0-9-]", " ", clean).strip()
            clean = re.sub("\s+", " ", clean)

            # clean = re.sub("\s-\s", "-", clean)
            clean_sentences.append(clean)
        return clean_sentences, dirty_sentences

    @staticmethod
    def sentence_to_vector(sentences: str) -> list:
        """
        Function:     Convert a sentence into a vector via embedding

        Inputs:       A list of sentences (list)

        Returns:      A list of sentences in the form of vectors (list)
                      Shape = (number of sentences x 768)
        """
        sentence_vectors = []
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if sentence_length != 0:
                vect = SentenceRankerUtil.embedding_model.encode(
                    sentence)
            else:
                vect = np.zeros((768, ))
            sentence_vectors.append(vect)
        return sentence_vectors

    @staticmethod
    def matrix_preparation(sentence_vectors: list) -> np.ndarray:
        """
        Function:     Prepares the sentence ranking matrix for optimisation

        Inputs:       Sentence vectors

        Returns:      Matrix
        """
        num_sentences = len(sentence_vectors)
        similarity_matrix = np.zeros([num_sentences, num_sentences])

        for i in range(num_sentences):
            for j in range(num_sentences):
                if i != j:
                    similarity_matrix[i][j] = cosine_similarity(
                        sentence_vectors[i].reshape(1, 768), sentence_vectors[j].reshape(1, 768))[0, 0]
        return similarity_matrix

    @staticmethod
    def run_text_rank(text: str) -> dict:
        """
        Function:     Extracts the top 'number' sentences from the article

        Inputs:       The article itself (string)
                      Number; the top however many sentences (integer)

        Returns:      The top `number` sentences with their scores
        """
        # parsing the article into an nlp instance
        doc = nlp(text)

        # extracting sentences out of the article
        sentences = [i.text.strip() for i in doc.sents]

        # converting sentences into vectors
        sentence_vectors = SentenceRankerUtil.sentence_to_vector(sentences)

        # creating a similarity matrix
        sim_mat = SentenceRankerUtil.matrix_preparation(sentence_vectors)

        # applying the page-ranking algorithm to the matrix
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

        # Extracting the sentence and scores
        sentence_scores = {i: [s, scores[i]] for i, s in enumerate(sentences)}
        return sentence_scores


def rank_sentences(text: str, n_sentences: int = 3) -> dict:
    sentence_scores = SentenceRankerUtil.run_text_rank(text)
    sentence_scores = OrderedDict(
        sorted(
            sentence_scores.items(),
            key=lambda x: x[1][1],
            reverse=True
        )
    )
    result = {}
    for i, (sentence_number, (sentence, score)) in enumerate(sentence_scores.items()):
        if pd.isna(score):
            score = 0
        result[i] = {'sentence': sentence, 'score': int(score*1000)/1000}
        if i + 1 >= n_sentences:
            break
    return result
