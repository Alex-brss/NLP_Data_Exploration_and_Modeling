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

# ---------- Parameters ---------- #

max_length_coef = 1.5
min_length_coef = 2

# ---------- Functions ---------- #

# ---------- Loading the dataset ---------- #

df = pd.read_csv('current_yelp_reviews.csv')[0:15]

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
        new = gensim.utils.simple_preprocess(text, deacc=False)
        tokens.append(new)

    return tokens

# Preprocessing function
def preprocessing(text):
    # Corrected spelling
    corrected_text = str(TextBlob(text).correct())

    # Translation
    cleaned_text = translate_text(str(corrected_text))

    return cleaned_text

def tokenised_text(text):
    # Lower case
    lower_text = text.lower()

    # Lemmatization & Tokenisation
    tokens = tokenisation(lower_text)

    return tokens

# Apply preprocessing and tokenisation
df['cleaned_text'] = df['text'].apply(preprocessing)
df['tokens'] = df['cleaned_text'].apply(tokenised_text)

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

pyLDAvis.enable_notebook(local=True)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds='mmds', R=10)
pyLDAvis.display(vis)

# ---------- Summarization ---------- #

# summariser = pipeline("summarization", model="facebook/bart-large-cnn")
# summarised_text = df['text'].apply(lambda x: summariser(x, max_length=round(len(x)/max_length_coef), min_length=round(len(x)/min_length_coef), do_sample=False))
# df['summarised_text'] = summarised_text.apply(lambda x: x[0]['summary_text'])

# ---------- Streamlit application ---------- #

def resize_image(image_path, width, height):
    image = Image.open(image_path)
    resized_image = image.resize((width, height))
    return resized_image

def main():
    st.set_page_config(page_title="Gastonomy", page_icon="🍽️", layout="wide")
    
    st.sidebar.markdown("Select a city and a restaurant to generate a review.")
    city = st.sidebar.selectbox("City", sorted(df['location'].unique()))
    
    # Dictionary mapping city names to image filenames
    city_images = {
        'New Orleans': 'resources/new-orleans.jpg',
        'New York City': 'resources/new-york.jpg',
        'Chicago': 'resources/chicago.jpg',
        'Los Angeles': 'resources/los-angeles.jpg',
        'San Francisco': 'resources/san-francisco.jpg',
        'Philadelphia': 'resources/philadelphia.jpg',
        'Las Vegas': 'resources/las-vegas.jpg',
        'Houston': 'resources/houston.jpg',
        'Phoenix': 'resources/phoenix.jpg',
        'Miami': 'resources/miami.jpg'
    }
    
    # Display image based on selected city
    if city in city_images:
        image_filename = city_images[city]
        resized_image = resize_image(image_filename, 1920, 1080)
        st.image(resized_image, caption=city)
    else:
        st.write("Image not found for selected city.")
    
if __name__ == '__main__':
    print('running')
    # main()