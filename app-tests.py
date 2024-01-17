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

df = pd.read_csv('current_yelp_reviews.csv')

print(df)

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
    main()