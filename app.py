from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load precomputed data
movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movie_list = movies['title'].values

def fetch_poster(movie_name):
    """Return placeholder image for all movies"""
    return "https://via.placeholder.com/500x750?text=No+Image"

def recommend(movie):
    import numpy as np
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list_sim = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = []
    for i in movie_list_sim:
        movie_title = movies.iloc[i[0]].title
        recommended_movies.append({
            "title": movie_title,
            "poster": fetch_poster(movie_title)
        })
    return recommended_movies

@app.route('/')
def home():
    return render_template('index.html', movie_list=movie_list)

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie_name = request.form['movie_name']
    recommendations = recommend(movie_name)
    return render_template('index.html', movie_list=movie_list, recommendations=recommendations, selected_movie=movie_name)

if __name__ == '__main__':
    app.run(debug=True)
