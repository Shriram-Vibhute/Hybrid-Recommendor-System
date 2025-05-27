import pandas as pd
import numpy as np
import scipy
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_recommendation(song_name: str, artist_name: str, df_songs: pd.DataFrame, interaction_matrix: scipy.sparse.csr_matrix) -> pd.DataFrame:
    # Lowercase the song name and artist name
    song_name = song_name.lower()
    artist_name = artist_name.lower()
    
    # Fetch the row from songs data
    ind = df_songs.loc[(df_songs["name"] == song_name) & (df_songs["artist"] == artist_name)].index[0]
    
    # Fetch the input vector
    input_array = interaction_matrix[ind]
    
    # Get similarity scores
    similarity_scores = cosine_similarity(input_array, interaction_matrix)

    # Returning similaritie scores
    return similarity_scores

def recommended_songs(df_songs: pd.DataFrame, similarity_scores: np.ndarray, k: int = 10):
    idx = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    print(df_songs.loc[idx, ["name", "artist"]])

def main():
    # Creating Paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    filtered_data_url = data_path / "interim" / "filtered_songs.csv"

    # load the data
    df_songs = pd.read_csv(filtered_data_url)
    interaction_matrix = scipy.sparse.load_npz(data_path / "processed" / "interaction_matrix.npz")

    # Calculate Similarity Scores
    recommendations = collaborative_recommendation(
        song_name = "stickwitu",
        artist_name = "the pussycat dolls",
        df_songs = df_songs,
        interaction_matrix = interaction_matrix
    )

    print(recommended_songs(df_songs = df_songs, similarity_scores = recommendations))
    
if __name__ == "__main__":
    main()
