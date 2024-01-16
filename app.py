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

load_dotenv()

# Parameters

max_length_coef = 1.5
min_length_coef = 2

# Functions

# Loading the dataset

df = pd.read_csv('current_yelp_reviews.csv')

df.drop_duplicates(inplace=True)
df.dropna(subset=['text', 'rating', 'location'], inplace=True)

# Preprocessing

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].astype(str)

def contains_chinese(text):
    return bool(re.search('[\u4e00-\u9fff]', text))

def translate_text(text):
    if contains_chinese(text):
        return translator(text)[0]['translation_text']
    else:
        return text

def preprocessing(text):
    # Corrected spelling
    corrected_text = TextBlob(text).correct()

    # Translation
    translated_text = translate_text(str(corrected_text))

    # Lower case
    lowercase_text = translated_text.lower()

    # Tokenization
    tokenised_text = word_tokenize(lowercase_text)

    # Remove punctuation and stop words
    cleaned_text = [word for word in tokenised_text if word.isalpha() and word not in stop_words]

    return cleaned_text

df['tokens'] = df['text'].apply(preprocessing)

# Summarization

summariser = pipeline("summarization", model="facebook/bart-large-cnn")
summarised_text = df['text'].apply(lambda x: summariser(x, max_length=round(len(x)/max_length_coef), min_length=round(len(x)/min_length_coef), do_sample=False))
df['summarised_text'] = summarised_text.apply(lambda x: x[0]['summary_text'])

# Streamlit application

def main():
    st.set_page_config(page_title="Gastonomy", page_icon="🇫🇷🍽️", layout="wide")

    st.markdown("Gastonomy, your very own french restaurant review generator. Now available in main US cities!")
    
    st.sidebar.subheader("Generate a review")

    st.sidebar.markdown("Select a city and a restaurant to generate a review.")

    city = st.sidebar.selectbox("City", ['New Orleans', 'New York City', 'Chicago', 'Los Angeles', 'San Francisco', 'Philadelphia', 'Las Vegas', 'Houston', 'Phoenix', 'Miami'])

    restaurant = st.sidebar.selectbox("Restaurant", df[df['location'] == city]['name'].unique())

    st.sidebar.markdown("Select the number of reviews to generate.")

    num_reviews = st.sidebar.slider("Number of reviews", 1, 10)

    st.sidebar.markdown("Select the length of the reviews to generate.")

    review_length = st.sidebar.slider("Review length", 1, 10)

    st.sidebar.markdown("Select the rating of the reviews to generate.")

    rating = st.sidebar.slider("Rating", 1, 5)

    st.sidebar.markdown("Select the type of reviews to generate.")

    review_type = st.sidebar.selectbox("Review type", ['Positive', 'Negative'])
    

if __name__ == '__main__':
    main()