import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("static\processed_data_movie.csv")
lemmatizer = WordNetLemmatizer()
tfidf = CountVectorizer()
vec = tfidf.fit_transform(df['short_summary'])
vec = pd.DataFrame(vec.toarray())

def recommend_movie(inp_array):
    similarities = cosine_similarity(inp_array, vec)
    most_similar = np.argsort(similarities)[:,-5:].reshape(5)
    return most_similar

def input_process(inp):
    text = inp.lower()
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    text = [lemmatizer.lemmatize(x) for x in word_tokenize(text) if x not in stopwords.words('english')]
    text = ' '.join(text)
    return tfidf.transform([text])
