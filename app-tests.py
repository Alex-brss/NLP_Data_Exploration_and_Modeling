# Imports

import nltk
import requests
import time
import pandas as pd
import os
import re
import gensim
import gensim.corpora as corpora
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import streamlit as st

from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from collections import Counter
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from dotenv import find_dotenv, load_dotenv
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Word2Vec
from sklearn.manifold import TSNE
from PIL import Image

load_dotenv()

df = pd.read_csv('yelp_reviews.csv')

# ---------- Highlighting frequent words ---------- #

# Word Frequency Analysis
all_words = [word for tokens in df['tokens'] for word in tokens]
word_freq = Counter(all_words)

# N-gram Analysis
bigrams = ngrams(all_words, 2)
bigram_freq = Counter(bigrams)

# Tri-gram Analysis
trigrams = ngrams(all_words, 3)
trigram_freq = Counter(trigrams)

# ---------- Topic Modelling ---------- #

# We convert the tokens into tuples where we'll have the word index (its placement on the map) and its frequency
id2word = corpora.Dictionary(df['tokens'])
corpus = [id2word.doc2bow(text) for text in df['tokens']]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=10,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

import streamlit as st

def display_topics(model, num_topics):
    for i in range(num_topics):
        words = model.show_topic(i)
        st.write(f"Topic {i+1}:")
        st.write(", ".join([word[0] for word in words]))

st.title("Restaurant Review Analysis")

st.header("Review Topics")
display_topics(lda_model, 10)