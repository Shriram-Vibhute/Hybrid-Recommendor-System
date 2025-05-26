import numpy as np
import pandas as pd
import scipy
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity_scores(input_vector: scipy.sparse.csr_matrix, songs_processed: scipy.sparse.csr_matrix):
    # Calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, songs_processed)
    return similarity_scores

def content_recommendation(song_name: str, artist_name: str, df_songs: pd.DataFrame, transformed_data: scipy.sparse.csr_matrix, k: int = 10):

    # convert song name and artist name to lowercase
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    # Filter out the song from data
    song_index = df_songs.loc[(df_songs["name"] == song_name) & (df_songs["artist"] == artist_name)].index[0]

    # Generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)

    # Calculate similarity scores
    similarity_scores = calculate_similarity_scores(input_vector, transformed_data)
    print(np.sort(similarity_scores).ravel()[-k-1:][::-1])

    # Get the top k songs
    top_k_songs_indexes = np.argsort(similarity_scores.ravel())[-k-1:][::-1]

    # Get the top k songs names
    top_k_songs_names = df_songs.iloc[top_k_songs_indexes]

    # Print the top k songs
    top_k_list = top_k_songs_names[['name','artist','spotify_preview_url']].reset_index(drop=True)
    return top_k_list

def main():
    # Creating Paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    songs_data_url = data_path / "interim" / "songs_interim.csv"

    # load the data
    df_songs = pd.read_csv(songs_data_url)
    df_songs_transformed = scipy.sparse.load_npz(data_path / "processed" / "songs_processed.npz")

    # Calculate Similarity Scores
    recommendations = content_recommendation(
        song_name = "seven nation army",
        artist_name = "the white stripes",
        df_songs = df_songs,
        transformed_data = df_songs_transformed,
        k = 10
    )

    print(recommendations)
    
if __name__ == "__main__":
    main()