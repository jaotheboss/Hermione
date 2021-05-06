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
