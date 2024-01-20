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

from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
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

load_dotenv()
np.random.seed(42)

# ---------- Functions ---------- #

# ---------- Loading the dataset ---------- #

df = pd.read_csv('yelp_reviews.csv')

df.drop_duplicates(inplace=True)
df.dropna(subset=['text', 'rating', 'location'], inplace=True)

# ---------- Preprocessing ---------- #

# Translation pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].astype(str)

# Check if text contains Chinese characters
def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

# Translation function (from Chinese to English)
def translate_text(text):
    if contains_chinese(text):
        return translator(text)[0]['translation_text']
    else:
        return text
    
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

# Preprocessing function
def preprocessing(text):
    # Corrected spelling on lower case text
    corrected_text = str(TextBlob(text.lower()).correct())

    # Translation
    cleaned_text = translate_text(str(corrected_text))

    return cleaned_text

# Apply preprocessing and tokenisation
df['cleaned_text'] = df['text'].apply(preprocessing)
df['tokens'] = tokenisation(df['cleaned_text'])

# # Summarisation

# max_length_coef = 1.5
# min_length_coef = 2

# summariser = pipeline("summarization", model="facebook/bart-large-cnn")
# summarised_text = df['text'].apply(lambda x: summariser(x, max_length=round(len(x)/max_length_coef), min_length=round(len(x)/min_length_coef), do_sample=False))
# df['summarised_text'] = summarised_text.apply(lambda x: x[0]['summary_text'])

# ---------- Highlighting frequent words ---------- #

review_frequent_words = {}

def get_frequency(restaurant_id):

    # Word Frequency Analysis
    all_words = [word for tokens in df[df['restaurant_id'] == restaurant_id]['tokens'] for word in tokens]
    word_freq = Counter(all_words)

    # N-gram Analysis
    bigrams = ngrams(all_words, 2)
    bigram_freq = Counter(bigrams)

    # Tri-gram Analysis
    trigrams = ngrams(all_words, 3)
    trigram_freq = Counter(trigrams)

    return [word_freq, bigram_freq, trigram_freq]

for restaurant_id in df['restaurant_id']:
    review_frequent_words[restaurant_id] = get_frequency(restaurant_id)

review_frequent_words_df = pd.DataFrame.from_dict(review_frequent_words, orient='index', columns=['word_freq', 'bigram_freq', 'trigram_freq'])
# review_frequent_words_df['word_freq'] = review_frequent_words_df['word_freq'].apply(lambda x: dict(sorted(x.items(), key=lambda item: item[1], reverse=True)))
review_frequent_words_df

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

def get_topic_distribution(lda_model, bow):
    return lda_model.get_document_topics(bow, minimum_probability=0)

df['topic_distribution'] = [get_topic_distribution(lda_model, corpus[i]) for i in range(len(df))]

def get_top_topics(topic_distribution, num_topics=5):
    # Sort the topics by probability and select the top ones
    return sorted(topic_distribution, key=lambda x: x[1], reverse=True)[:num_topics]

df['top_topics'] = df['topic_distribution'].apply(lambda x: get_top_topics(x, 11 - 1))

def label_topics(topic_list, lda_model):
    labels = []
    for topic_id, _ in topic_list:
        # Get the top words in the topic
        words = lda_model.show_topic(topic_id, 5)
        # Create a label (e.g., by joining the top words)
        label = [word for word, prob in words]
        labels.append(label)
    return labels

def topicise(labels, label_dict):
    topics = []

    for topic_list in labels:
        for key, value in label_dict.items():
            if set(topic_list) == set(value):
                topics.append(key)

    return topics

label_dict = {
    'Quality of Food & Service' : ['service', 'food', 'restaurant', 'good', 'great'],
    'French Dining Experience' : ['dinner', 'meal', 'french', 'reservation', 'little'],
    'Atmosphere' : ['speak', 'dining', 'menu', 'experience', 'soup'],
    'Price' : ['course', 'table', 'thing', 'life', 'party'],
    'Special Occasions' : ['birthday', 'time', 'family', 'really', 'warm'],
    'Ambience' : ['experience', 'overall', 'kiss', 'attentive', 'fantastic'],
    'Dining Experience' : ['experience', 'overall', 'kiss', 'attentive', 'fantastic'],
    'Staff' : ['year', 'last', 'time', 'first', 'second'],
    'Menu' : ['atmosphere', 'area', 'bit', 'high', 'mummy'],
    'Drinks' : ['way', 'incredible', 'class', 'wall', 'mood'] 
}

df['top_topic_labels'] = df['top_topics'].apply(lambda x: label_topics(x, lda_model))
df['topics'] = df['top_topic_labels'].apply(lambda x: topicise(x, label_dict))
df.drop(columns=['topic_distribution', 'top_topics'], inplace=True)

# ---------- Sentiment Analysis ---------- #

# ---------- Word2Vec ---------- #

# We train our own word2vec model

word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("yelp_reviews.model")

