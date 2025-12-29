import pandas as pd
import nltk
import time
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from redisclient import RedisCache

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")


class MovieRecommender:
    def __init__(self) -> None:
        self.movies_df = pd.read_csv("movies.csv")
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

        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )

        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["content"])

        self.cache = RedisCache()

    def analyze_emotion(self, text: str) -> str:
        result = self.emotion_classifier(text)[0][0]
        return result["label"].lower()

    def get_recommendations(self, user_text: str, num_recommendations: int = 5):
        normalized_text = user_text.lower().strip()

        cached_emotion = self.cache.get_emotion(normalized_text)
        if cached_emotion:
            emotion = cached_emotion
            cache_hit = True
        else:
            emotion = self.analyze_emotion(user_text)
            self.cache.set_emotion(normalized_text, emotion)
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


if __name__ == "__main__":
    recommender = MovieRecommender()

    while True:
        text = input("Enter your experience (or 'quit'): ").strip()
        if text.lower() == "quit":
            print("Bye")
            break

        start = time.time()
        movies, emotion, cached = recommender.get_recommendations(text)
        end = time.time()

        print("\nMovies:", movies)
        print("Emotion:", emotion)
        print("Cache hit:", cached)
        print("Time taken:", round(end - start, 4), "seconds\n")
