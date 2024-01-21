
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
import torch
import numpy as np
import tensorflow as tf
import gensim.downloader as api
import tensorflow_hub as hub
import nlpaug.augmenter.word as naw
import random
import warnings
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from langchain import PromptTemplate, LLMChain
from dotenv import find_dotenv, load_dotenv
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Word2Vec, KeyedVectors
from sklearn.manifold import TSNE
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorboard.plugins import projector
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import LineSentence
from tensorboard.plugins import projector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Model, Sequential
from scipy.spatial.distance import euclidean, cosine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from nlpaug.util import Action
from scipy import spatial

df = pd.read_csv("/Users/alexandrecogordan/Documents/ESILV/Ongoing/Machine Learning For NLP/Project 2/NLP_Data_Exploration_and_Modeling/yelp_reviews.csv")

stop_words = set(stopwords.words('english'))

# Lemmatisation & Tokenisation function
def tokenisation(reviews, allowed_postags=["NOUN", "ADJ", "VERBS", "ADV"]):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    reviews_out = []
    tokens = []

    for review in reviews:
        doc = nlp(review) 
        reviews_out.append(" ".join([token.lemma_ for token in doc if token.pos_ in allowed_postags and token.lemma_ not in stop_words]))
    
    for text in reviews_out:
        new = gensim.utils.simple_preprocess(text, deacc=False) # We do not remove the accent marks because we deem them important for French restaurants reviews
        tokens.append(new)

    return tokens

df['tokens'] = tokenisation(df['cleaned_text'])

model = Word2Vec.load("word2vec.model")

def review_to_vector(review, model):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def semantic_search(query, model, reviews, topn=10):
    query_vector = review_to_vector(query, model)

    # Calculate similarity between query and each review
    similarities = []
    for review in reviews:
        review_vector = review_to_vector(review, model)
        similarity = 1 - spatial.distance.cosine(query_vector, review_vector)
        similarities.append((review, similarity))

    # Sort reviews by similarity
    sorted_reviews = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Return the topn most similar reviews
    return sorted_reviews[:topn]

user_query = "I wish to have a nice dinner with my family"

search_results = semantic_search(user_query, model, df['cleaned_text'].tolist())

for review, similarity in search_results:
    filtered_df_three = df[df['cleaned_text'].apply(lambda x: review in x)]
    print(filtered_df_three)