import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout = "centered")
st.title("Movie Recommender")

df=pd.read_csv("data/tmdb_5000_movies.csv")
with open("models/embeddings.pkl", "rb") as f:
    embeddings=pickle.load(f)

def recommend(title):
    title=title.lower()
    matches=df[df['title'].str.lower()==title]
    if matches.empty:
        return []
    idx=matches.index[0]
    score=cosine_similarity([embeddings[idx]], embeddings)[0]
    top_movie_idx=np.argsort(score)[-6:-1][::-1]
    return df['title'].iloc[top_movie_idx].tolist()

movie=st.text_input("Enter movie name")

if st.button("Recommend"):
    if movie.strip()=="":
        st.warning("Please enter movie name")
    else:
        results=recommend(movie)
        if results:
            st.subheader("Recommended movies")
            for r in results:
                st.write("🎥", r)
        else:
            st.warning("Movie not found. Check spelling")