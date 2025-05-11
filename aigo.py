import streamlit as st
import torch
import torch.nn as nn
import fastai
from fastai.learner import load_learner

# Disable warnings that might come from PyTorch/fastai version compatibility
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_model(model_path):
    """Load the model with caching to improve performance"""
    try:
        return load_learner(model_path, cpu=True)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    st.title("Movie Recommendation 🎬")
    st.write("Enter your favorite movie title to get recommendations!")
    
    # Load model
    model_path = "model.pkl"
    learn = load_model(model_path)
    
    if learn is None:
        st.error("Failed to load the recommendation model. Please check the model file.")
        return
    
    # Get all movie titles
    all_movie_titles = learn.dls.classes['original_title']
    
    # Create text input for movie title
    movie_title = st.text_input("Enter a movie title:")
    
    if movie_title:
        try:
            movie_classes = learn.dls.classes['original_title'] 
            movie_title_lower = movie_title.lower()
            
            # Find the movie index
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
            st.exception(e)

if __name__ == "__main__":
    main()
