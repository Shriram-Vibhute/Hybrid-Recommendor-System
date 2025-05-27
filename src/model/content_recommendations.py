import numpy as np
import pandas as pd
import scipy
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

def content_recommendation(song_name: str, artist_name: str, df_songs: pd.DataFrame, transformed_data: scipy.sparse.csr_matrix):

    # Convert song name and artist name to lowercase
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    # Filter out the song from data
    song_index = df_songs.loc[(df_songs["name"] == song_name) & (df_songs["artist"] == artist_name)].index[0]

    # Generate the input vector
    input_vector = transformed_data[song_index].reshape(1,-1)

    # Calculate similarity scores
    content_similarity_scores = cosine_similarity(input_vector, transformed_data)

    # Returning similarities scores
    return content_similarity_scores

def recommended_songs(df_songs: pd.DataFrame, similarity_scores: np.ndarray, k: int = 10):
    idx = np.argsort(similarity_scores.ravel())[::-1][:k+1]
    print(df_songs.loc[idx, ["name", "artist"]])

def main():
    # Creating Paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    songs_data_url = data_path / "interim" / "filtered_songs.csv"

    # load the data
    df_songs = pd.read_csv(songs_data_url)
    df_songs_transformed = scipy.sparse.load_npz(data_path / "processed" / "songs_processed.npz")

    # Calculate Similarity Scores
    recommendations = content_recommendation(
        song_name = "stickwitu",
        artist_name = "the pussycat dolls",
        df_songs = df_songs,
        transformed_data = df_songs_transformed,
    )
    
    recommended_songs(df_songs = df_songs, similarity_scores = recommendations, k = 10)
if __name__ == "__main__":
    main()