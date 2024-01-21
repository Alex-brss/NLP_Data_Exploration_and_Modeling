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

warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

# Load environment variables
load_dotenv()

# ---------- Functions ---------- #

# Preprocessing function
def preprocessing(text):
    # Corrected spelling on lower case text
    corrected_text = str(TextBlob(text.lower()).correct())

    return corrected_text

# ---------- Loading the dataset ---------- #

df = pd.read_csv('yelp_reviews.csv')

df.drop_duplicates(inplace=True)
df.dropna(subset=['text', 'rating', 'location'], inplace=True)

# ---------- Preprocessing ---------- #

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

# w2v_model = gensim.models.keyedvectors.KeyedVectors.load("word2vec.model")

# def semantic_search(query_word, model, topn=10):
#     query_vector = model.wv[query_word]
#     all_words = model.wv.index_to_key

#     # Calculate cosine distance between query and all other words
#     distances = {word: cosine(query_vector, model.wv[word]) for word in all_words}
    
#     # Sort words by distance (lower is more similar)
#     sorted_words = sorted(distances, key=distances.get)

#     # Return the topn closest words
#     return sorted_words[:topn]

# Example usage
# search_results = semantic_search('wine', w2v_model)

# ---------- Application Functions ---------- #

model = Word2Vec.load("word2vec.model")
summariser = pipeline("summarization", model="facebook/bart-large-cnn")

# ------------- To delete

model = api.load('glove-twitter-50') # Change to 200 when we have the time

def vectorize_model(sent, model):
    vector_size = model.vector_size
    model_res = np.zeros(vector_size)
    ctr = 1
    for word in sent:
        if word in model:
            ctr += 1
            model_res += model[word]
    model_res = model_res/ctr
    return model_res

df['vectors'] = df['tokens'].apply(vectorize_model, model=model)

# -------------

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
    
        ## 

        # Find the row in the df dataframe that corresponds to the review
        row = df[df['review'] == review]
        restaurant_id = row['restaurant_id'].values[0]
        
        # Check if the restaurant_id is already in the similarities list
        if not any(restaurant_id == r[0]['restaurant_id'] for r in similarities):
            similarities.append((row, similarity))

        ##

        # Replace with this if this doesn't work
            
        # similarities.append((review, similarity))

    # Sort reviews by similarity
    sorted_reviews = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Return the topn most similar reviews
    return sorted_reviews[:topn]

def classify_review(review_text, pipeline):
    predicted_label = pipeline.predict([review_text])
    return predicted_label

# ---------- Application ---------- #

# Other features like sentiment analysis for example have been directly added in the dataframe to preserve computational time. Please see the notebook for more details.

import streamlit as st
import joblib

pipeline = joblib.load('review_classification_pipeline.joblib')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="Gastonomy", page_icon="🍽️", layout="wide")
city = st.sidebar.selectbox("City", sorted(df['location'].unique()))

filtered_df_one = df[df['location'] == city]

st.title("Restaurant Review Analysis")

st.header("Quel sont les aspects les plus importants pour vous dans un restaurant?")
topics = st.multiselect("Choisissez vos aspects", sorted(label_dict.keys()))

filtered_df_two = filtered_df_one[filtered_df_one['topics'].apply(lambda x: all(topic in x for topic in topics))]

if topics:
    user_query = st.text_input("What are you looking for in a restaurant?")
    search_results = semantic_search(user_query, model, df['cleaned_text'].tolist())

    for review, similarity in search_results:
        filtered_df_three = pd.concat([filtered_df_two[filtered_df_two['cleaned_text'] == review], filtered_df_three])

    if user_query:
        st.write("Here is the list of the top 10 restaurants you might like. Choose one to get its general feeling and a summary about the general feeling people have about it based on its reviews.")
        selected_restaurant = st.selectbox("Select an item:", filtered_df_three['restaurant_id'])

        # Give a summary of the restaurant
        st.subheader("Review summary for this restaurant")
        
        selected_reviews = df[df['restaurant_id'] == selected_restaurant]['cleaned_text'].tolist()
        combined_reviews = ' '.join(selected_reviews)

        # Generate summary
        summary = summarizer(combined_reviews, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        st.write(summary)


        # We get the reviews for the selected restaurant
        final_reviews = filtered_df_three[filtered_df_three['restaurant_id'] == selected_restaurant]['cleaned_text'].tolist()

        if selected_restaurant:
            if st.button("Classify Reviews"):

                sentiments = []

                for review in final_reviews:
                    sentiments.append(classify_review(review))

                sentiment = np.mean(sentiments)

                # Display the result
                if sentiment > 2.5:
                    st.write("Most reviews for this restaurant are positive. You should pay them a visit!")
                elif sentiment >= 1.5 and sentiment <= 2.5:
                    st.write("Most reviews for this restaurant neutral. You shouldn't be worried but you shouldn't expect anything either.")
                else:
                    st.write("Most reviews for this restaurant are negative. You should visit at your own risk.")