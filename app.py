import sys
import os
import numpy as np
import pandas as pd
import pathlib
import streamlit as st
from scipy.sparse import load_npz

# Add the src directory to Python path
sys.path.append(str(pathlib.Path(__file__).parent))

# Now import your modules
from src.model.hybrid_recommendations import HybridRecommenderSystem

# Importing necessary paths
home_path = pathlib.Path(__file__).resolve().parent
data_path = home_path / "data"
df_songs = pd.read_csv(filepath_or_buffer = data_path / "interim" / "filtered_songs.csv")
transformed_data = load_npz(file = data_path / "processed" / "songs_processed.npz")
interaction_matrix = load_npz(file = data_path / "processed" / "interaction_matrix.npz") 

# Setting the structure of webapp
st.set_page_config(page_title="Hybrid Recommender System", layout="wide")

# Title
st.title('Welcome to the Spotify Song Recommender!')

# Image
from PIL import Image
image = Image.open(home_path / "assets" / "spotify.jpg")
st.image(image)

# Description
st.subheader('Enter the name of a song and the recommender will suggest similar songs 🎵🎧')

# Song Input
song_name = st.selectbox(label = 'Enter a song name:', options = df_songs['name'])

# Artist Input
song_artists = df_songs[df_songs['name'] == song_name]
artist_name = st.selectbox(label = 'Enter the artist name:', options = song_artists['artist'])

# k recommndations
k = st.selectbox(label = 'How many recommendations do you want?', options = [5,10,15,20], index=1)

# Type of Recommendation
recommendation_type = st.selectbox(label = "Type of Recommendation", options = ["Content Base Recommendations", "Collaburative Base Recommendation", "Hybrid Recommendation"])

if recommendation_type == "Hybrid Recommendation":
    # diversity slider
    diversity = st.slider(
        label="Diversity in Recommendations",
        min_value=1,
        max_value=9,
        value=5,
        step=1
    )
    content_based_weight = 1 - (diversity / 10)
    # plot a bar graph
    chart_data = pd.DataFrame({
        "type" : ["Personalized", "Diverse"],
        "ratio": [10 - diversity, diversity]
    })
    st.bar_chart(chart_data,x="type",y="ratio")

# Button
if st.button('Get Recommendations'):
    recommender = HybridRecommenderSystem(
        number_of_recommendations = k,
        weight_content_based = 0 if recommendation_type == "Collaburative Base Recommendation" else 1 if recommendation_type == "Content Base Recommendations" else content_based_weight
    ) 
    st.write(f'{recommendation_type} for', f"'{song_name}' by '{artist_name}'")

    # get the recommendations
    recommendations = recommender.give_recommendations(
        song_name = song_name,
        artist_name = artist_name,
        songs_data = df_songs,
        transformed_matrix = transformed_data,
        interaction_matrix = interaction_matrix)

    # Display Recommendations
    for ind , recommendation in recommendations.iterrows():
        song_name = recommendation['name'].title()
        artist_name = recommendation['artist'].title()
        
        if ind == 0:
            st.markdown("## Currently Playing")
            st.markdown(f"#### **{song_name}** by **{artist_name}**")
            st.audio(recommendation['spotify_preview_url'])
            st.write('---')
        elif ind == 1:   
            st.markdown("### Next Up 🎵")
            st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
            st.audio(recommendation['spotify_preview_url'])
            st.write('---')
        else:
            st.markdown(f"#### {ind}. **{song_name}** by **{artist_name}**")
            st.audio(recommendation['spotify_preview_url'])
            st.write('---')
