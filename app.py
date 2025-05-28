# app.py
import streamlit as st
from recommender import MovieRecommender

st.set_page_config(page_title="Mood-Based Movie Recommender", layout="centered")

st.title("🎬 Mood-Based Movie Recommender")
st.write("Tell us how you're feeling, and we'll suggest some movies that match your mood!")

# Initialize the recommender
recommender = MovieRecommender()

user_input = st.text_area("How are you feeling today?", height=150)

if st.button("🎥 Recommend Movies"):
    if user_input.strip():
        with st.spinner("Analyzing emotion and fetching recommendations..."):
            recommendations, emotion = recommender.get_recommendations(user_input)

        st.success(f"🧠 Detected Emotion: {emotion.capitalize()}")

        st.markdown("### 🎯 Recommended Movies:")
        for i, movie in enumerate(recommendations, 1):
            st.markdown(f"**{i}. {movie['title']}**  \n*Genres:* {movie['genres']}")
    else:
        st.warning("Please enter how you're feeling to get recommendations.")
