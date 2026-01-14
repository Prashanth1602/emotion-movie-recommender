import logging
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from redisclient import RedisCache

logger = logging.getLogger(__name__)

class MovieRecommender:
    def __init__(self, csv_path: str, cache_client=None) -> None:
        logger.info(f"Initializing MovieRecommender with CSV: {csv_path}")
        self.movies_df = pd.read_csv(csv_path)
        self.movies_df = self.movies_df[["Title", "Genre", "Description"]].fillna(" ")
        self.movies_df.columns = ["title", "genres", "overview"]

        self.emotion_genre_mapping = {  
            "joy": ["Comedy", "Animation", "Family"],
            "sadness": ["Drama", "Romance"],
            "anger": ["Action", "Crime"],
            "fear": ["Horror", "Mystery", "Thriller"],
            "surprise": ["Sci-Fi", "Adventure", "Fantasy"],
            "neutral": ["Documentary", "Biography"],
            "love": ["Romance", "Drama"],
            "disgust": ["Horror", "Thriller"],
        }

        self.movies_df["content"] = (
            self.movies_df["genres"] + " " + self.movies_df["overview"]
        )

        logger.info("Loading emotion classification model...")
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )

        logger.info("Vectorizing movie content...")
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["content"])

        self.cache = cache_client or RedisCache()
        logger.info("MovieRecommender initialization complete.")

    def analyze_emotion(self, text: str) -> str:
        result = self.emotion_classifier(text)[0][0]
        return result["label"].lower()

    def get_recommendations(self, user_text: str, num_recommendations: int = 5):
        normalized_text = user_text.lower().strip()

        emotion = None
        cache_hit = False

        try:
            emotion = self.cache.get_emotion(normalized_text)
            if emotion:
                cache_hit = True
                logger.info(f"Cache hit for '{normalized_text}'")
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")

        if not emotion:
            emotion = self.analyze_emotion(user_text)
            try:
                self.cache.set_emotion(normalized_text, emotion)
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
            cache_hit = False

        relevant_genres = self.emotion_genre_mapping.get(emotion, [])

        user_vector = self.tfidf.transform([user_text])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)[0]

        recommendations = []
        for score, movie in zip(similarities, self.movies_df.itertuples()):
            genres = [g.strip() for g in str(movie.genres).split("|")]
            if any(g in relevant_genres for g in genres):
                recommendations.append((movie.title, score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        movie_list = [m[0] for m in recommendations[:num_recommendations]]

        return movie_list, emotion, cache_hit


