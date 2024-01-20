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
import torch.nn.functional as F

# Set random seed
np.random.seed(42)

# Load environment variables
load_dotenv()

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

# ---------- Semantic Search ---------- #

w2v_model = gensim.models.keyedvectors.KeyedVectors.load("word2vec.model")

def semantic_search(query_word, model, topn=10):
    query_vector = model.wv[query_word]
    all_words = model.wv.index_to_key

    # Calculate cosine distance between query and all other words
    distances = {word: cosine(query_vector, model.wv[word]) for word in all_words}
    
    # Sort words by distance (lower is more similar)
    sorted_words = sorted(distances, key=distances.get)

    # Return the topn closest words
    return sorted_words[:topn]

# Example usage
search_results = semantic_search('wine', w2v_model)

# ---------- Application ---------- #

# Other features like sentiment analysis for example have been directly added in the dataframe to preserve computational time. Please see the notebook for more details.

import streamlit as st

st.set_page_config(page_title="Gastonomy", page_icon="🍽️", layout="wide")
city = st.sidebar.selectbox("City", sorted(df['location'].unique()))

# def resize_image(image_path, width, height):
#     image = Image.open(image_path)
#     resized_image = image.resize((width, height))
#     return resized_image

# # Dictionary mapping city names to image filenames
# city_images = {
#     'New Orleans': 'resources/new-orleans.jpg',
#     'New York City': 'resources/new-york.jpg',
#     'Chicago': 'resources/chicago.jpg',
#     'Los Angeles': 'resources/los-angeles.jpg',
#     'San Francisco': 'resources/san-francisco.jpg',
#     'Philadelphia': 'resources/philadelphia.jpg',
#     'Las Vegas': 'resources/las-vegas.jpg',
#     'Houston': 'resources/houston.jpg',
#     'Phoenix': 'resources/phoenix.jpg',
#     'Miami': 'resources/miami.jpg'
# }

# # Display image based on selected city
# if city in city_images:
#     image_filename = city_images[city]
#     resized_image = resize_image(image_filename, 1920, 1080)
#     st.image(resized_image, caption=city)
# else:
#     st.write("Image not found for selected city.")

st.title("Restaurant Review Analysis")

st.header("Quel sont les aspects les plus importants pour vous dans un restaurant?")
topics = st.multiselect("Choisissez vos aspects", sorted(label_dict.keys()))
