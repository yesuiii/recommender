import streamlit as st
import torch
import torch.nn as nn
import fastai
from fastai.learner import load_learner


model_path = "model.pkl"
learn = load_learner(model_path, cpu=True)
all_movie_titles = learn.dls.classes['original_title']

def main():
    st.title("Movie Recommendation 🎬")
    st.write("Enter your favorite movie title to get recommendations!")
    movie_title = st.text_input("Enter a movie title:")

    if movie_title:
        try:
            movie_classes = learn.dls.classes['original_title'] 
            movie_title_lower = movie_title.lower()
            idx = -1
            for i, title in enumerate(movie_classes):
                if title.lower() == movie_title_lower:
                    idx = i
                    break

            if idx != -1:
                movie_factors = learn.model.i_weight.weight
                distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors[idx][None])
                idxs = distances.argsort(descending=True)[1:6]
                recommendations = [movie_classes[i] for i in idxs]

                st.write("### Top 5 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.write(f"Movie '{movie_title}' not found in the dataset. Showing similar recommendations based on popular movies:")

                movie_factors = learn.model.i_weight.weight

                distances = nn.CosineSimilarity(dim=1)(movie_factors, movie_factors.mean(dim=0, keepdim=True))
                idxs = distances.argsort(descending=True)[1:6]  

                recommendations = [movie_classes[i] for i in idxs]

                st.write("### Top 5 Recommendations (even if movie is not found):")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
