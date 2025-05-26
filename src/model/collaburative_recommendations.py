import pandas as pd
import numpy as np
import scipy
import pathlib
from sklearn.metrics.pairwise import cosine_similarity

def collaborative_recommendation(song_name: str, artist_name: str, track_ids: str, df_songs: pd.DataFrame, interaction_matrix: scipy.sparse.csr_matrix, k: int = 5) -> pd.DataFrame:
    # Lowercase the song name and artist name
    song_name = song_name.lower()
    artist_name = artist_name.lower()
    
    # fetch the row from songs data
    song_row = df_songs.loc[(df_songs["name"] == song_name) & (df_songs["artist"] == artist_name)]

    # track_id of input song
    input_track_id = song_row['track_id'].values.item()
  
    # index value of track_id
    ind = np.where(track_ids == input_track_id)[0].item()
    
    # fetch the input vector
    input_array = interaction_matrix[ind]
    
    # get similarity scores
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    
    # index values of recommendations
    recommendation_indices = np.argsort(similarity_scores.ravel())[-k-1:][::-1]
    
    # get top k recommendations
    recommendation_track_ids = track_ids[recommendation_indices]
    
    # get top scores
    top_scores = np.sort(similarity_scores.ravel())[-k-1:][::-1]
    
    # get the songs from data and print
    scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),
                            "score":top_scores})
    
    top_k_songs = (
        df_songs
        .loc[df_songs["track_id"].isin(recommendation_track_ids)]
        .merge(scores_df,on="track_id")
        .sort_values(by="score",ascending=False)
        .drop(columns=["track_id","score"])
        .reset_index(drop=True)
    )
    
    return top_k_songs

def main():
    # Creating Paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    songs_data_url = data_path / "interim" / "songs_interim.csv"
    users_data_url = data_path / "interim" / "users_interim.csv"
    unique_track_ids_url = data_path / "processed" / "unique_track_ids.npy"

    # load the data
    df_songs = pd.read_csv(songs_data_url)
    df_users = pd.read_csv(users_data_url)
    unique_track_ids = np.load(unique_track_ids_url, allow_pickle = True)
    interaction_matrix = scipy.sparse.load_npz(data_path / "processed" / "interaction_matrix.npz")

    # Calculate Similarity Scores
    recommendations = collaborative_recommendation(
        song_name = "take me out",
        artist_name = "franz ferdinand",
        track_ids = unique_track_ids,
        df_songs = df_songs,
        interaction_matrix = interaction_matrix,
        k = 10
    )

    print(recommendations)
    
if __name__ == "__main__":
    main()
