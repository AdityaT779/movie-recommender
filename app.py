import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process


st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("Film Recommendation System")


df = pd.read_csv("data/tmdb_5000_movies.csv")

#Transformed using Sentence Transformer on other page, directly loading here
with open("models/overview_embeddings.pkl", "rb") as f:
    overview_embeddings = pickle.load(f)

with open("models/genres_embeddings.pkl", "rb") as f:
    genres_embeddings = pickle.load(f)

with open("models/keywords_embeddings.pkl", "rb") as f:
    keywords_embeddings = pickle.load(f)

def closest_title(user_input, titles, threshold=70):
    match = process.extractOne(
        user_input,
        titles,
        score_cutoff=threshold
    )
    if match:
        return match[0]
    return None

def recommend(movie_title):

    titles = df["title"].values #converts column of movie names into a numpy array
    closest = closest_title(movie_title, titles)
    if closest is None:
        return []

    idx = df[df["title"] == closest].index[0]

    overview_scores = cosine_similarity(
        [overview_embeddings[idx]],       #cosine similiarity uses 2D list
        overview_embeddings
    )[0]

    genre_scores = cosine_similarity(
        [genres_embeddings[idx]],
        genres_embeddings
    )[0]

    keyword_scores = cosine_similarity(
        [keywords_embeddings[idx]],
        keywords_embeddings
    )[0]

    w_plot = 0.5
    w_genre = 0.3
    w_keyword = 0.2

    final_scores = (
        w_plot * overview_scores +
        w_genre * genre_scores +
        w_keyword * keyword_scores
    )

    top_idx = np.argsort(final_scores)[-6:-1][::-1]

    results = []

    for i in top_idx:
        plot_contrib = w_plot * overview_scores[i]
        genre_contrib = w_genre * genre_scores[i]
        keyword_contrib = w_keyword * keyword_scores[i]

        total = plot_contrib + genre_contrib + keyword_contrib

        explanation = {
            "title": df["title"].iloc[i],
            "plot_pct": (plot_contrib / total) * 100,
            "genre_pct": (genre_contrib / total) * 100,
            "keyword_pct": (keyword_contrib / total) * 100
        }

        results.append(explanation)

    return results

def next_movie():
    if st.session_state.i < len(st.session_state.recs) - 1:
        st.session_state.i += 1


def prev_movie():
    if st.session_state.i > 0:
        st.session_state.i -= 1

movie = st.text_input("Enter a movie name:")


if st.button("Recommend"):

    if movie.strip() == "":
        st.warning("Please enter a movie name.")

    else:
        recs = recommend(movie)

        if not recs:
            st.warning("Movie not found. Check spelling.")

        else:
            st.session_state.recs = recs
            st.session_state.i = 0


if "recs" in st.session_state:

    recs = st.session_state.recs
    idx = st.session_state.i

    current = recs[idx]

    st.subheader(f"🎬 Recommendation {idx + 1}")
    st.write("**Movie:**", current["title"])

    st.write("Why this was recommended:")
    st.write(f"📖 Plot similarity: **{current['plot_pct']:.1f}%**")
    st.write(f"🎭 Genre similarity: **{current['genre_pct']:.1f}%**")
    st.write(f"🔑 Keyword similarity: **{current['keyword_pct']:.1f}%**")


    col1, col2 = st.columns(2)

    with col1:
        st.button("⬅ Previous", on_click=prev_movie)

    with col2:
        st.button("Next ➡", on_click=next_movie)


    st.write("---")
    st.subheader("Top 5 Recommendations (Quick View)")

    for r in recs:
        st.write("🎥", r["title"])