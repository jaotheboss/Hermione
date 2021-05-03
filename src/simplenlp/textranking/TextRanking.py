"""
How to use:
instance = TextRanker() or SentenceRanker()
instance.extract(text, ...)
"""

# NLP Modules
from sentence_transformers import SentenceTransformer
import re                                               # text cleaning
from spacy.lang.en.stop_words import STOP_WORDS         # importing a list of stop words
import spacy                                            # main nlp module
nlp = spacy.load('en_core_web_sm')

# Network/Graphing Modules
import networkx as nx                                   # matrix manipulation
from sklearn.metrics.pairwise import cosine_similarity  # similarity computation

# General Modules
import numpy as np                                      # array manipulation
import pandas as pd
from collections import OrderedDict                     # for data storage

class WordRanker():
       """
       Main class that ranks each keyword of a sentence based on Google's
       Page Ranking algorithm

       It can take in a single sentence to an entire paragraph.
       """
       def __init__(self):
              self.damping_coef = 0.85                  # recommended
              self.min_diff = 1e-5                      # convergence threshold
              self.steps = 10                           # number of iterations
              self.candidate_pos = ['NOUN', 'PROPN']    # which kinds of words to keep for analysis

              self.verbose = False

              # carries the importance value of each word
              self.node_weight = None                   # save keywords and their weights

       def set_steps(self, new_steps: int) -> None:
              self.steps = new_steps
              if self.verbose:
                     print('Iteration steps have been updated!')
       
       def set_threshold(self, new_threshold: int) -> None:
              self.min_diff = new_threshold
              if self.verbose:
                     print('New threshold has been set!')

       def edit_candidate_pos(self, candidate_pos: str) -> None:
              candidate_pos = list(candidate_pos)
              for i in candidate_pos:
                     if i not in self.candidate_pos:
                            self.candidate_pos.append(i)
              if self.verbose:
                     print('Candidate POS has been updated!')

       def set_stopwords(self) -> None:
              """
              Function:     Set stopwords for the nlp module

              Input:        None

              Returns:      Confirmation (string)
              """
              for word in STOP_WORDS:
                     lexeme = nlp.vocab[word]
                     lexeme.is_stop = True
              if self.verbose:
                     print('Stopwords has been set!')

       def add_stopwords(self, stopwords: str) -> None:
              """
              Function:     Add stopwords to the additional list

              Input:        An iterable or a string of stopword(s)

              Return:       Confirmation (string)
              """
              # ensure they're all in an iterable
              stopwords = set(list(stopwords))

              # add new words into the nlp vocabulary
              for new in stopwords:
                     lexeme = nlp.vocab[new]
                     lexeme.is_stop = True
              
              # add everything to the stop_words set
              STOP_WORDS = STOP_WORDS.union(stopwords)

              # show confirmation of completion
              if self.verbose:
                     print('New stopwords added!')

       def segment_sentences(self, nlp_doc: object) -> list:
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
                            if token.pos_ in self.candidate_pos and token.is_stop is False:
                                   key_words.append(token.text.lower())
                     sentence_key_words.append(key_words)
              if self.verbose:
                     print('Sentences have been filtered for their keywords!')
              return sentence_key_words

       def get_vocabulary(self, segmented_sentences: list) -> OrderedDict:
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
              if self.verbose:
                     print('Vocabulary dictionary has been created and indexed!')
              return vocab

       def build_token_pairs(self, segmented_sentences: list, window_size: int) -> list:
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
              if self.verbose:
                     print('Token pairs have been created!')
              return token_pairs
       
       def build_keyword_matrix(self, vocab: OrderedDict, token_pairs: list) -> np.ndarray:
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
                      dtype = 'float'
              )
              for word1, word2 in token_pairs:
                     i, j = vocab[word1], vocab[word2]
                     matrix[i][j] = 1
              
              # Symmetrizing the matrix
              matrix = matrix + matrix.T - np.diag(matrix.diagonal())

              # Normalizing the columns of the matrix
              norm = np.sum(matrix, axis = 0)
              matrix = np.divide(
                     matrix,
                     norm,
                     where = norm != 0
              )
              if self.verbose:
                     print('Keyword matrix has been created and ready for PageRanking!')
              return matrix

       def run_text_rank(self, text: str, window_size: int) -> None:
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
              self.set_stopwords()

              # parse text with spaCy and create doc object
              doc = nlp(text)

              # segment the sentences for keywords
              sentences = self.segment_sentences(doc)

              # build the dictionary out of the keywords only
              vocab = self.get_vocabulary(sentences)

              # get the token pairs from the windows out of the keywords only
              token_pairs = self.build_token_pairs(sentences, window_size)

              # initialising the matrix and the pagerank values
              matrix = self.build_keyword_matrix(vocab, token_pairs)
              pagerank_values = np.array([1] * len(vocab))

              # iteration of the weights via the pageranking algorithm
              previous_pagerank_sum = 0
              for epoch in range(self.steps):
                     # execute pageranking algorithm
                     pagerank_values = (1 - self.damping_coef) + self.damping_coef * np.dot(matrix, pagerank_values)
                     current_pagerank_sum = sum(pagerank_values)
                     if abs(previous_pagerank_sum - current_pagerank_sum) < self.min_diff:
                            break
                     else:
                            previous_pagerank_sum = current_pagerank_sum
              
              # extract weight for each node
              node_weight = dict()
              for word, index in vocab.items():
                     node_weight[word] = pagerank_values[index]
              
              self.node_weight = node_weight

       def rank(self, text: str, n_words: int = 10, window_size: int = 5) -> list:
              """
              Function:     Simplifies everything and combines all methods
                            and algorithm to extract the ranked keywords
                            from text

              Inputs:       The text itself (string)
                            Number of words to be extracted (int)
                            Window size for window analysis on text

              Returns:      A list of keywords with their pagerank values
              """
              # update the weights
              self.run_text_rank(text, window_size)
              if self.verbose:
                     print('Weights have been optimised!')

              # sort the weights
              node_weight = OrderedDict(
                     sorted(
                            self.node_weight.items(), 
                            key = lambda x: x[1], 
                            reverse = True
                     )
              )

              # extract the keywords based on their weights
              result = [['Rank', 'Keyword', 'PageRank Value']]
              for i, (key, value) in enumerate(node_weight.items()):
                     if pd.isna(value):
                            value = 0
                     result.append([i + 1, key, int(value*100000)/100000])
                     if i + 1 >= n_words:
                            break
              return result

class SentenceRanker():
       """
       This is the main class that ranks each sentence within an article
       """
       def __init__(self):
              self.verbose = False

              self.embedding_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
              self.max_seq_len = 128
              self.embedding_model.max_seq_length = self.max_seq_len

       def change_seq_len(self, length: int) -> None:
              self.max_seq_len = length
              self.embedding_model.max_seq_length = length

       def doc_to_sentence(self, nlp_doc: object) -> [list, list]:
              """
              Function:     Extracts the sentences from a nlp doc object and 
                            cleans them before returning

              Inputs:       A nlp doc object from spaCy

              Returns:      Cleaned sentences (string)
              """
              dirty_sentences = [i.text.strip() for i in nlp_doc.sents]
              clean_sentences = []
              for dirt in dirty_sentences:
                     clean = " ".join([i.lower() for i in dirt])

                     # removing anything that's not a word
                     clean = re.sub("[^a-zA-Z0-9-]", " ", clean).strip()
                     clean = re.sub("\s+", " ", clean)

                     # clean = re.sub("\s-\s", "-", clean)
                     clean_sentences.append(clean)
              return clean_sentences, dirty_sentences
       
       def sentence_to_vector(self, sentences: str) -> list:
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
                            vect = self.embedding_model.encode(sentence)
                     else:
                            vect = np.zeros((768, ))
                     sentence_vectors.append(vect)
              return sentence_vectors
       
       def matrix_preparation(self, sent_vects: list, num_sentences: int) -> np.ndarray:
              """
              Function:     Prepares the sentence ranking matrix for optimisation

              Inputs:       Sentence vectors

              Returns:      Matrix
              """
              similarity_matrix = np.zeros([num_sentences, num_sentences])

              for i in range(num_sentences):
                     for j in range(num_sentences):
                            if i != j:
                                   similarity_matrix[i][j] = cosine_similarity(sent_vects[i].reshape(1, 768), sent_vects[j].reshape(1, 768))[0, 0]
              return similarity_matrix

       def rank(self, article: str, n_sentences: int = 3) -> list:
              """
              Function:     Extracts the top 'number' sentences from the article

              Inputs:       The article itself (string)
                            Number; the top however many sentences (integer)

              Returns:      The top `number` sentences with their scores
              """
              # parsing the article into an nlp instance
              doc = nlp(article)

              # extracting sentences out of the article
              sentences = [i.text.strip() for i in doc.sents]
              n_sentences = len(sentences)

              # converting sentences into vectors
              sentence_vectors = self.sentence_to_vector(sentences)

              # creating a similarity matrix 
              sim_mat = self.matrix_preparation(sentence_vectors, n_sentences)

              # applying the page-ranking algorithm to the matrix
              nx_graph = nx.from_numpy_array(sim_mat)
              scores = nx.pagerank(nx_graph)

              # Extracting the sentence and scores
              ranked_sentences = sorted([[i, s, scores[i]] for i, s in enumerate(sentences)], key = lambda x: x[2], reverse=True)
              ranked_sentences = [['Sentence Number', 'Text Sentence', 'PageRank Score']] + ranked_sentences

              return ranked_sentences[:n_sentences]

