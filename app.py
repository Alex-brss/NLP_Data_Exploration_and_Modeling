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
import streamlit as st
import joblib

from torch.utils.tensorboard import SummaryWriter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from textblob import TextBlob
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
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

api_key = os.getenv('YELP_SEARCH_API_KEY')
headers = {'Authorization': 'Bearer ' + api_key}\

# ---------- Adding link retrieval from yelp ---------- #

def get_reviews(id):
    url = f"https://api.yelp.com/v3/businesses/{id}/reviews?sort_by=yelp_sort&limit=2"
    response = requests.get(url, headers=headers)
    reviews_data = response.json()
    
    try:
        return reviews_data['reviews']
    except:
        review = "No reviews for this restaurant"
        
    return []

# ---------- Preprocessing ---------- #

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
stop_words = set(stopwords.words('english'))

# Check if text contains Chinese characters
def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

# Translation function (from Chinese to English)
def translate_text(text):
    if contains_chinese(text):
        return translator(text)[0]['translation_text']
    else:
        return text
    
# We remove special characters/punctuation whilst keeping characters with accents since their meaning is important in French restaurants reviews
def remove_special_characters(text):
    pattern = r"[^\w\séèàçêôœ]"
    
    # We substitute matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text
    
# Preprocessing function - which regroups all of the previous steps too
def preprocessing(text):
    # Translation
    translated_text = translate_text(str(text))

    # We remove special characters/punctuation
    corrected_text = remove_special_characters(translated_text)

    # Corrected spelling on lower case text
    cleaned_text = str(TextBlob(corrected_text.lower()).correct())

    return cleaned_text

# ---------- Loading the dataset ---------- #

df = pd.read_csv('analysed_reviews.csv')

df.drop_duplicates(inplace=True)
df.dropna(subset=['text', 'sentiment', 'location'], inplace=True)

locations = sorted(df['location'].unique())
locations.append('All')

# ---------- Topics ---------- #

label_dict = {
    'Overall Experience': ['restaurant', 'time', 'dinner', 'first', 'friend', 'wine', 'meal', 'friendly', 'branch'],
    'Food & Service': ['food', 'service', 'good', 'great', 'alliance', 'excellent', 'staff', 'location', 'witness'],
    'Decor': ['place', 'nice', 'beautiful', 'table', 'area', 'space', 'bread', 'wall'],
    'French Aspect': ['french', 'dish', 'favorite', 'spot', 'afternoon', 'lunch', 'onion', 'soup'],
    'Coffee & Tea': ['tea', 'lovely', 'cafe', 'worth'],
    'Drinks': ['reservation', 'drink', 'bar', 'still', 'pm', 'holiday', 'center'],
    'Gastronomical Aspect': ['menu', 'star', 'old', 'yelp', 'minute', 'weekend', 'ally', 'appearance', 'gastro', 'item'],
    'Parking': ['attentive', 'second', 'park'],
    'Atmosphere': ['atmosphere', 'excited', 'week', 'door', 'couple'],
    'Events & Parties': ['party', 'private', 'open', 'venue']
}

# ---------- Application Functions ---------- #

model = Word2Vec.load("word2vec.model")

pipe = pipeline("text2text-generation", model="mrm8488/t5-base-finetuned-summarize-news")

# In case GloVe is used

# model = api.load('glove-twitter-50')

# def vectorize_model(sent, model):
#     vector_size = model.vector_size
#     model_res = np.zeros(vector_size)
#     ctr = 1
#     for word in sent:
#         if word in model:
#             ctr += 1
#             model_res += model[word]
#     model_res = model_res/ctr
#     return model_res

# df['vectors'] = df['tokens'].apply(vectorize_model, model=model)

# def review_to_vector(review, model):
#     words = review.split()
#     word_vectors = [model.wv[word] for word in words if word in model]
#     if len(word_vectors) == 0:
#         return np.zeros(model.vector_size)
#     return np.mean(word_vectors, axis=0)

# def semantic_search(query, model, reviews):
#     query_vector = review_to_vector(query, model)

#     # Calculate similarity between query and each review
#     similarities = []
    
#     restaurant_set = set()

#     for review in reviews:
#         review_vector = review_to_vector(review, model)
#         similarity = 1 - spatial.distance.cosine(query_vector, review_vector)

#         row = df[df['cleaned_text'] == review]
#         restaurant_id = row['restaurant_id'].values[0]
        
#         if restaurant_id not in restaurant_set:
#             similarities.append((review, similarity))
#             restaurant_set.add(restaurant_id)
            
#     # Sort reviews by similarity
#     sorted_reviews = sorted(similarities, key=lambda x: x[1], reverse=True)

#     # Return the topn most similar reviews
#     return sorted_reviews

# In case Word2Vec is used

def review_to_vector(review, model):
    words = review.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]  # Adjusted for model.wv
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def semantic_search(query, model, reviews, df):  # Added df as an argument
    query_vector = review_to_vector(query, model)

    similarities = []
    restaurant_set = set()

    for review in reviews:
        review_vector = review_to_vector(review, model)
        if np.all(review_vector == 0):
            continue

        similarity = 1 - spatial.distance.cosine(query_vector, review_vector)
        if np.isnan(similarity):
            continue

        row = df[df['cleaned_text'] == review]
        if row.empty:
            continue 

        restaurant_id = row['restaurant_id'].values[0]
        
        if restaurant_id not in restaurant_set:
            similarities.append((review, similarity))
            restaurant_set.add(restaurant_id)
            
    sorted_reviews = sorted(similarities, key=lambda x: x[1], reverse=True)
    return sorted_reviews

def classify_review(review_text, review_pipeline):
    predicted_label = review_pipeline.predict([review_text])
    return predicted_label

# ---------- Application ---------- #

# Other features like sentiment analysis for example have been directly added in the dataframe to preserve computational time. Please see the notebook for more details.

review_pipeline = joblib.load('review_classification_pipeline.joblib')

st.set_page_config(page_title="Gastonomy", page_icon="🍽️", layout="wide")

city = st.sidebar.selectbox("City", locations)

if city != 'All':
    df = df[df['location'] == city]

st.title("Restaurant Review Analysis 👨‍🍳")

st.header("Restaurant opinion")
restaurant_link = st.text_input("Predict the overall sentiment of a restaurant. Give us the restaurant's link. ")

if restaurant_link:
    restaurant_id = restaurant_link.split("/biz/")[1].split("?")[0]

    final_reviews = get_reviews(restaurant_id)
    combined_reviews = '. '.join([preprocessing(review['text']) for review in final_reviews])

    st.subheader("Review summary for this restaurant")

    prompt = "summarise: " + combined_reviews
    summary = pipe(prompt, do_sample=False)[0]['generated_text']
    st.write(str(summary).capitalize())
    
    sentiments = []

    for review in final_reviews:
        sentiments.append(classify_review(preprocessing(review['text']), review_pipeline))

    sentiment = np.mean(sentiments)

    sentiment_text = "Most reviews for this restaurant are negative. You should visit at your own risk."

    stars_rating = "⭐ "

    # Display the result
    if sentiment > 1.3:
        sentiment_text = "Most reviews for this restaurant are positive. You should pay them a visit!"
        stars_rating = stars_rating + "⭐ ⭐ ⭐ ⭐"
    elif sentiment >= 0.7 and sentiment <= 1.3:
        sentiment_text = "Most reviews for this restaurant neutral. You shouldn't be worried but you shouldn't expect anything either."
        stars_rating = stars_rating + "⭐ ⭐"

    st.subheader("Overall sentiment for this restaurant")
    st.write(sentiment_text)
    st.write("Our rating: ", stars_rating)

st.header("Restaurant finder")
st.subheader("What aspects are most important to you?")

topics = st.multiselect("Choose your aspects", sorted(label_dict.keys()))

filtered_df_two = df[df['topics'].apply(lambda x: all(topic in x for topic in topics))]

if topics:
    user_query = st.text_input("What are you looking for in a restaurant?")
    search_results = semantic_search(user_query, model, df['cleaned_text'].tolist(), df)

    filtered_df_three = pd.DataFrame(columns=df.columns)

    for review, similarity in search_results:
        filtered_df_three = pd.concat([filtered_df_two[filtered_df_two['cleaned_text'] == review], filtered_df_three])

    if user_query:
        st.write("Here are some restaurants you might like. Choose one to get its general feeling and a summary about the general feeling people have about it based on its reviews.")
        selected_restaurant = st.selectbox("Select an item:", filtered_df_three['business_name'].unique())

        if selected_restaurant:
            # Give a summary of the restaurant
            st.subheader("Review summary for this restaurant")
            
            selected_reviews = df[df['business_name'] == selected_restaurant]['cleaned_text'].tolist()
            combined_reviews = '. '.join(selected_reviews)

            # Summary using a prompt
            prompt = "summarise: " + combined_reviews
            summary = pipe(prompt, do_sample=False)[0]['generated_text']

            # We get the reviews for the selected restaurant
            final_reviews = filtered_df_three[filtered_df_three['business_name'] == selected_restaurant]['cleaned_text'].tolist()

            sentiments = []

            for review in final_reviews:
                sentiments.append(classify_review(review, review_pipeline))

            sentiment = np.mean(sentiments)

            sentiment_text = "Most reviews for this restaurant are negative. You should visit at your own risk."

            stars_rating = "⭐ "

            # Display the result
            if sentiment > 1.3:
                sentiment_text = "Most reviews for this restaurant are positive. You should pay them a visit!"
                stars_rating = stars_rating + "⭐ ⭐ ⭐ ⭐"
            elif sentiment >= 0.7 and sentiment <= 1.3:
                sentiment_text = "Most reviews for this restaurant neutral. You shouldn't be worried but you shouldn't expect anything either."
                stars_rating = stars_rating + "⭐ ⭐"

            st.write(str(summary).capitalize())

            st.subheader("Overall sentiment for this restaurant")
            st.write(sentiment_text)
            st.write("Our rating: ", stars_rating)

            restaurant_id = filtered_df_three[filtered_df_three['business_name'] == selected_restaurant]['restaurant_id'].values[0]
            restaurant_link = f"[Link to the restaurant](https://www.yelp.com/biz/{restaurant_id})"
            st.markdown("Link of the restaurant: " + restaurant_link, unsafe_allow_html=True)

# IMPORTANT: To run the application, please use the following command: streamlit run app.py