# recommender.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

nltk.download('stopwords')

class MovieRecommender:
    def __init__(self, csv_path='movies.csv'):
        self.movies_df = pd.read_csv(csv_path)
        self.movies_df = self.movies_df[['Title', 'Genre', 'Description']].fillna('')
        self.movies_df.columns = ['title', 'genres', 'overview']

        self.emotion_genre_mapping = {
            'joy': ['Comedy', 'Animation', 'Family'],
            'sadness': ['Drama', 'Romance'],
            'anger': ['Action', 'Crime'],
            'fear': ['Horror', 'Mystery', 'Thriller'],
            'surprise': ['Sci-Fi', 'Adventure', 'Fantasy'],
            'neutral': ['Documentary', 'Biography'],
            'love': ['Romance', 'Drama'],
            'disgust': ['Horror', 'Thriller'],
        }

        self.movies_df['content'] = self.movies_df['genres'] + ' ' + self.movies_df['overview']
        self.stop_words = set(stopwords.words('english'))

        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=False
        )

    def analyze_emotion(self, text):
        result = self.emotion_classifier(text)[0]
        return result['label'].lower()

    def get_recommendations(self, user_text, num_recommendations=5):
        emotion = self.analyze_emotion(user_text)
        relevant_genres = self.emotion_genre_mapping.get(emotion, [])

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        tfidf = TfidfVectorizer(stop_words='english')
        combined_texts = self.movies_df['content'].tolist() + [user_text]
        tfidf_matrix = tfidf.fit_transform(combined_texts)

        user_vector = tfidf_matrix[-1]
        movie_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(user_vector, movie_vectors)[0]

        recommended_movies = []
        for score, movie in zip(similarities, self.movies_df.itertuples()):
            genres = [g.strip() for g in movie.genres.split('|')]
            if any(g in relevant_genres for g in genres):
                recommended_movies.append({
                    'title': movie.title,
                    'genres': movie.genres,
                    'similarity': score
                })

        recommended_movies.sort(key=lambda x: x['similarity'], reverse=True)
        return recommended_movies[:num_recommendations], emotion
