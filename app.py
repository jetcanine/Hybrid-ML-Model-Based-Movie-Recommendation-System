# app.py
from flask import Flask, session, redirect, url_for, request, render_template
from pymongo import MongoClient
import urllib
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tensorflow import keras
from huggingface_hub import hf_hub_download
from urllib.request import urlretrieve
from zipfile import ZipFile
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from prometheus_client import generate_latest, make_wsgi_app, Counter, Histogram, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time
import psutil
import threading

app = Flask(__name__)
app.secret_key = os.urandom(24)



# MongoDB Configuration
MONGO_URI = "mongodb+srv://admin:admin@cluster0.ipm8e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
# MONGO_URI = "mongodb+srv://admin:" + urllib.parse.quote("admin") + "@cluster0.iftro.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client.movie_recommender
users_collection = db.users


# Constants
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
MODEL_HF_REPO = "svhozt/profile_based_recommendation_model.keras"
MODEL_FILENAME = "profile_based_recommendation_model.keras"
NUM_CLUSTERS = 20


# Prometheus Metrics for monitoring
RECOMMENDATION_COUNTER = Counter(
    'recommendations_total', 
    'Total number of recommendations made'
)
FEEDBACK_COUNTER = Counter(
    'feedback_total',
    'User feedback counts',
    ['feedback_type']
)
RECOMMENDATION_LATENCY = Histogram(
    'recommendation_latency_seconds',
    'Time taken to generate recommendations'
)
CPU_GAUGE = Gauge('cpu_usage_percent', 'Current CPU usage')
MEMORY_GAUGE = Gauge('memory_usage_percent', 'Current memory usage')

# CORS support for dashboard figured it out in 15 mins :)
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

# system monitoring thread
def monitor_system():
    while True:
        CPU_GAUGE.set(psutil.cpu_percent())
        MEMORY_GAUGE.set(psutil.virtual_memory().percent)
        time.sleep(5)

#  monitoring thread
monitor_thread = threading.Thread(target=monitor_system)
monitor_thread.daemon = True
monitor_thread.start()


# Data Ingestion Component
class DataIngestion:
    
    def __init__(self, dataset_url, dataset_dir="dataset"):
        self.dataset_url = dataset_url
        self.dataset_dir = dataset_dir

    def download_data(self):

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            urlretrieve(self.dataset_url, "dataset.zip")
            ZipFile("dataset.zip", "r").extractall(self.dataset_dir)

        users = pd.read_csv(f"{self.dataset_dir}/ml-1m/users.dat", sep="::", engine='python',
                            names=["user_id", "sex", "age_group", "occupation", "zip_code"])
        ratings = pd.read_csv(f"{self.dataset_dir}/ml-1m/ratings.dat", sep="::", engine='python',
                              names=["user_id", "movie_id", "rating", "unix_timestamp"])
        movies = pd.read_csv(f"{self.dataset_dir}/ml-1m/movies.dat", sep="::", engine='python',
                             names=["movie_id", "title", "genres"], encoding="ISO-8859-1")

        return users, ratings, movies
    

class DataPreprocessor:
    def __init__(self, users, ratings, movies):
        self.users = users
        self.ratings = ratings
        self.movies = movies

    def clean_data(self):
        ratings_movies = self.ratings.merge(self.movies, on="movie_id", how="left")
        top_ratings = ratings_movies[ratings_movies['rating'] >= 4].copy()
        top_ratings['genres_list'] = top_ratings['genres'].str.split('|')
        exploded_genres = top_ratings.explode('genres_list')

        preferred_genres = (
            exploded_genres.groupby('user_id')['genres_list']
            .apply(lambda x: x.value_counts().index[0] if not x.value_counts().empty else None)
            .reset_index()
        )
        preferred_genres.columns = ['user_id', 'preferred_genres']
        user_profiles = self.users.merge(preferred_genres, on="user_id", how="left")
        user_profiles['preferred_genres'] = user_profiles['preferred_genres'].fillna('')
        return user_profiles, exploded_genres

    def prepare_clustering_data(self, user_profiles):
        demographic_data = pd.get_dummies(user_profiles[['sex', 'age_group', 'occupation']], drop_first=True)
        genres_data = user_profiles['preferred_genres'].str.get_dummies('|')
        clustering_data = pd.concat([user_profiles[['user_id']], demographic_data, genres_data], axis=1)
        clustering_data = clustering_data.set_index('user_id')
        return clustering_data
    
class ProfileGenerator:
    def __init__(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters

    def cluster_user_profiles(self, clustering_data):
        kmeans = KMeans(n_clusters=self.number_of_clusters, n_init=8, random_state=42)
        clustering_data['profile_id'] = kmeans.fit_predict(clustering_data)
        return clustering_data.reset_index(), kmeans
    
class RecommendationSystem:
    def __init__(self):
        self.model = None
        self.recommended_movies = set()
        self.user_feedback = {}
        self.user_ratings = {}
        self.profile_weights = {}
        self.preferred_genres = []

    def load_model_from_huggingface(self):
        downloaded_model_path = hf_hub_download(repo_id=MODEL_HF_REPO, filename=MODEL_FILENAME)
        self.model = keras.models.load_model(downloaded_model_path)
        print("Model loaded successfully from Hugging Face.")

    def get_age_group(self, age):
        if age < 18: return 1
        elif 18 <= age <= 24: return 18
        elif 25 <= age <= 34: return 25
        elif 35 <= age <= 44: return 35
        elif 45 <= age <= 49: return 45
        elif 50 <= age <= 55: return 50
        else: return 56

    def add_user_to_cluster(self, user_df, clustering_data, kmeans_model):
        for col in clustering_data.columns:
            if col not in user_df.columns:
                user_df[col] = 0
        user_vector = user_df[clustering_data.drop(['user_id', 'profile_id'], axis=1).columns]
        profile_id = kmeans_model.predict(user_vector)[0]
        return profile_id
    
    def recommend_movie(self, profile_id, movies_df, top_n=5):
        start_time = time.time()
        if 'user_id' in session:
            user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
            self.preferred_genres = user.get('session_genres') or user.get('preferred_genres', [])
        else:
            self.preferred_genres = session.get('preferred_genres', [])

        movie_ids = movies_df["movie_id"].values
        max_embedding_index = 3882
        
        movies_with_genres = movies_df.copy()
        movies_with_genres['genres_list'] = movies_with_genres['genres'].fillna('').str.split('|')
        
        genre_matched_movies = movies_with_genres[
            movies_with_genres['genres_list'].apply(
                lambda x: any(genre in x for genre in self.preferred_genres)
            )
        ]
        
        valid_movie_ids = [
            mid for mid in genre_matched_movies['movie_id'].values if 
            mid <= max_embedding_index and 
            movies_with_genres[movies_with_genres['movie_id'] == mid]['title'].iloc[0] not in self.recommended_movies
        ]

        if not valid_movie_ids:
            return None

        profile_ids = np.full(shape=(len(valid_movie_ids)), fill_value=profile_id)
        movie_ids_input = np.array(valid_movie_ids)

        predictions = self.model.predict([profile_ids, movie_ids_input])
        
        if profile_id in self.profile_weights:
            predictions = predictions * self.profile_weights[profile_id]

        movie_predictions = pd.DataFrame({
            "movie_id": valid_movie_ids,
            "predicted_score": predictions.flatten()
        }).merge(genre_matched_movies, on="movie_id", how="left")
        
        movie_predictions['genre_match_score'] = movie_predictions['genres_list'].apply(
            lambda x: len(set(x) & set(self.preferred_genres)) / len(self.preferred_genres)
        )
        
        movie_predictions['final_score'] = movie_predictions['predicted_score'] * (1 + movie_predictions['genre_match_score'])
        top_movies = movie_predictions.sort_values(by="final_score", ascending=False).head(top_n)
        

        
        if not top_movies.empty:
            self.recommended_movies.add(top_movies.iloc[0]['title'])
        return top_movies

    def update_user_profile(self, profile_id, feedback, clustered_profiles):
        if profile_id not in self.profile_weights:
            self.profile_weights[profile_id] = 1.0

        if feedback:
            self.profile_weights[profile_id] *= 1.2
        else:
            self.profile_weights[profile_id] *= 0.7

        if self.profile_weights[profile_id] < 0.7:
            available_profiles = clustered_profiles['profile_id'].unique()
            available_profiles = available_profiles[available_profiles != profile_id]
            new_profile_id = np.random.choice(available_profiles)
            self.profile_weights[new_profile_id] = 1.0
            return new_profile_id
        return profile_id
    


# Initialize system components
def initialize_system():
    data_ingestion = DataIngestion(DATASET_URL)
    users, ratings, movies = data_ingestion.download_data()
    preprocessor = DataPreprocessor(users, ratings, movies)
    user_profiles, exploded_genres = preprocessor.clean_data()
    clustering_data = preprocessor.prepare_clustering_data(user_profiles)
    profile_generator = ProfileGenerator(NUM_CLUSTERS)
    clustered_profiles, kmeans_model = profile_generator.cluster_user_profiles(clustering_data)
    recommendation_system = RecommendationSystem()
    recommendation_system.load_model_from_huggingface()
    return {
        'movies': movies,
        'clustered_profiles': clustered_profiles,
        'kmeans_model': kmeans_model,
        'recommendation_system': recommendation_system
    }

system_components = initialize_system()

# Flask Routes
@app.before_request
def check_session():
    if request.endpoint != 'logout' and 'logged_in' in session:
        if not users_collection.find_one({'_id': ObjectId(session['user_id'])}):
            session.clear()
            return redirect(url_for('login'))

@app.route('/')
def index():
    session.clear()
    return redirect(url_for('login'))


#  Login Route 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = users_collection.find_one({'username': username})
        
        if user and check_password_hash(user['password'], password):
            session.update({
                'user_id': str(user['_id']),
                'logged_in': True,
                'profile_id': user['profile_id'],  # Set profile_id from DB
                'preferred_genres': user['preferred_genres'],
                'session_genres': user.get('session_genres', []),
                'age': user['age'],
                'sex': user['sex'],
                'occupation': user['occupation'],
                'feedback': {},
                'ratings': {}
            })
            return redirect(url_for('recommend'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        errors = []
        form_data = request.form.copy()
        
        # Validate required fields
        required_fields = {
            'username': 'Username is required',
            'password': 'Password is required',
            'age': 'Age is required',
            'gender': 'Gender is required',
            'occupation': 'Occupation is required'
        }
        
        # Check empty fields
        for field, message in required_fields.items():
            if not form_data.get(field):
                errors.append(message)
        
        # Check genre selection
        selected_genres = form_data.getlist('genres')
        if not selected_genres:
            errors.append("At least one genre must be selected")

        # Validate numerical fields
        try:
            age = int(form_data['age'])
            if not (1 <= age <= 100):
                errors.append("Age must be between 1-100")
        except ValueError:
            errors.append("Invalid age format")
        
        try:
            occupation = int(form_data['occupation'])
            if not (0 <= occupation <= 20):
                errors.append("Invalid occupation code")
        except ValueError:
            errors.append("Invalid occupation format")

        # Validate gender value
        if form_data['gender'] not in ['M', 'F']:
            errors.append("Invalid gender selection")

        # Check username uniqueness
        if not errors and users_collection.find_one({'username': form_data['username']}):
            errors.append("Username already exists")

        if errors:
            return render_template('register.html',
                                 genres=get_valid_genres(),
                                 errors=errors,
                                 form_data=form_data)

        # Original clustering logic remains unchanged
        user_df = pd.DataFrame({
            'sex_M': [1 if form_data['gender'] == 'M' else 0],
            'age_group': [RecommendationSystem().get_age_group(age)],
            'occupation': [occupation],
            **{genre: 0 for genre in get_valid_genres()}
        })
        
        for genre in selected_genres:
            if genre in user_df.columns:
                user_df[genre] = 1

        profile_id = system_components['recommendation_system'].add_user_to_cluster(
            user_df, 
            system_components['clustered_profiles'],
            system_components['kmeans_model']
        )

        # Create user document
        user_data = {
            'username': form_data['username'],
            'password': generate_password_hash(form_data['password']),
            'age': age,
            'sex': form_data['gender'],
            'occupation': occupation,
            'preferred_genres': selected_genres,
            'profile_id': int(profile_id),
            'session_genres': [],
            'feedback': []
        }
        users_collection.insert_one(user_data)
        return redirect(url_for('login'))
    
    return render_template('register.html', genres=get_valid_genres())

@app.route('/recommend')
def recommend():
    start_time = time.time()

    if 'feedback' not in session:
        session['feedback'] = {}
    if 'ratings' not in session:
        session['ratings'] = {}

    rec_system = system_components['recommendation_system']
    top_movies = rec_system.recommend_movie(
        session.get('profile_id', 0),
        system_components['movies'],
        top_n=1
    )
    
    if top_movies is None or top_movies.empty:
        return redirect(url_for('complete'))
    
    movie_data = top_movies.iloc[0].to_dict()
    session['current_movie'] = movie_data['title']
    RECOMMENDATION_LATENCY.observe(time.time() - start_time)
    RECOMMENDATION_COUNTER.inc()
    return render_template('recommendation.html', movie=movie_data)


@app.route('/feedback', methods=['POST'])
def handle_feedback():
    feedback = request.form.get('feedback') == 'yes'
    current_movie = session['current_movie']
    feedback_type = 'positive' if feedback else 'negative'
    FEEDBACK_COUNTER.labels(feedback_type).inc()
    # Record feedback in session
    session['feedback'][current_movie] = feedback
    session['ratings'][current_movie] = 5 if feedback else 1
    
    # Update database with feedback
    users_collection.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$push': {'feedback': {
            'movie': current_movie,
            'feedback': feedback,
            'timestamp': datetime.now()
        }}}
    )

    # Check recommendation accuracy periodically
    if len(session['feedback']) % 5 == 0:
        accuracy = sum(session['feedback'].values()) / len(session['feedback'])
        
        if accuracy < 0.6:  # Threshold for poor performance
            # Reset session genres to original preferences
            session['session_genres'] = session.get('preferred_genres', [])
            
            # Update database with accuracy status
            users_collection.update_one(
                {'_id': ObjectId(session['user_id'])},
                {'$set': {'low_accuracy_flag': True}}
            )
            
            return redirect(url_for('change_genres'))

    # Update recommendation count and check for exhaustion
    session['recommendation_count'] = session.get('recommendation_count', 0) + 1
    
    if session['recommendation_count'] >= 50:  # Max recommendations per session
        return redirect(url_for('complete'))

    return redirect(url_for('recommend'))

# app.py - Modify change_genres route
@app.route('/change-genres', methods=['GET','POST'])
def change_genres():
    if request.method == 'POST':
        session['preferred_genres'] = request.form.getlist('genres')
    
        if 'user_id' in session:
            users_collection.update_one(
                {'_id': ObjectId(session['user_id'])},
                {'$set': {
                    'session_genres': session['preferred_genres'],
                    # Clear previous temporary genres
                    'preferred_genres': users_collection.find_one(
                        {'_id': ObjectId(session['user_id'])}
                    )['preferred_genres']
                }}
            )
    
        # Reset recommendation state
        session['recommended'] = []
        session['feedback'] = {}
        session['ratings'] = {}
    
        return redirect(url_for('recommend'))
    return render_template('change_genres.html', genres=get_valid_genres())


@app.route('/logout')
def logout():
    if 'user_id' in session:
        user = users_collection.find_one({'_id': ObjectId(session['user_id'])})
        if user and user.get('session_genres'):
            return redirect(url_for('confirm_genre_change'))
    session.clear()
    return redirect(url_for('index'))

@app.route('/confirm-genre-change')
def confirm_genre_change():
    return render_template('confirm_genre_change.html')

@app.route('/complete')
def complete():
    save_changes = request.args.get('save_changes')
    
    if 'user_id' in session:
        user_id = ObjectId(session['user_id'])
        
        if save_changes == 'yes':
            # Save session genres to preferred genres
            users_collection.update_one(
                {'_id': user_id},
                {'$set': {
                    'preferred_genres': session.get('preferred_genres', []),
                    'session_genres': []
                }}
            )
        elif save_changes == 'no':
            # Clear session genres without saving
            users_collection.update_one(
                {'_id': user_id},
                {'$set': {'session_genres': []}}
            )
    
    # Clear all session data
    session.clear()
    return render_template('complete.html')

def get_valid_genres():
    return ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
           'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
           'Thriller', 'War', 'Western']


@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain; version=0.0.4'}


if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)