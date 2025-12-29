# Emotion-Based Movie Recommender

A high-performance, emotion-aware movie recommendation system built using **Python**, **Transformers**, **TF-IDF**, and **Redis caching**, optimized for low-latency inference

## Features

- Emotion detection from free-form user text using a pretrained Transformer model
- Emotion → Genre mapping for context-aware recommendations
- Content-based movie similarity using precomputed TF-IDF vectors
- Redis-based caching to eliminate redundant emotion inference
- Significant latency optimization through intelligent caching and vector reuse

## System Architecture

User Text -> Emotion Detection(Transformer) -> Redis Cache -> Genre Filtering -> TF-IDF Similarity -> Top-N Movie Recommendations

##  Performance Benchmarks

|        Scenario         |Avg Latency |
|-------------------------|------------|
| No cache, no precompute | ~0.9 sec   |
| TF-IDF precomputed only | ~0.3 sec   |
| Redis emotion cache + 
|                  TF-IDF | ~0.0085 sec|

> Achieved ~100× latency improvement using caching and precomputation.

## Tech Stack

- Python
- HuggingFace Transformers
- scikit-learn (TF-IDF, cosine similarity)
- Redis
- pandas, nltk

## Key Optimizations

- Precomputed TF-IDF matrix to avoid recomputation per request
- Redis TTL-based emotion caching to bypass repeated transformer inference
- Normalized cache keys for consistency and efficiency
- Clean separation of concerns for scalability

Setup and run command - 
> chmod +x run.sh
> ./run.sh
This script will:
Create a virtual environment (if not present)
Install required dependencies
Start Redis
Run the recommender application
