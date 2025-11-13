# movie_recommender_backend.py

import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
movies = pd.read_csv(r'C:\Users\Karkave D\.cache\kagglehub\datasets\tmdb\tmdb-movie-metadata\versions\2\tmdb_5000_movies.csv')
credits = pd.read_csv(r'C:\Users\Karkave D\.cache\kagglehub\datasets\tmdb\tmdb-movie-metadata\versions\2\tmdb_5000_credits.csv')

# Merge and preprocess
movies = movies.merge(credits, on='title')
movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]
movies.dropna(inplace=True)

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
new = movies.drop(columns=['overview','genres','keywords'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Vectorize and compute similarity
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vector)

# Save data
pickle.dump(new, open('movies.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))
