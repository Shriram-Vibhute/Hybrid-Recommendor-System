import numpy as np
import pandas as pd
import scipy
import pathlib

# Import from the same package
from .content_recommendations import content_recommendation
from .collaburative_recommendations import collaborative_recommendation

class HybridRecommenderSystem:
    
    def __init__(self, number_of_recommendations: int, weight_content_based: float):
        self.number_of_recommendations = number_of_recommendations
        self.weight_content_based = weight_content_based
        self.weight_collaborative = 1 - weight_content_based
        
    def __calculate_content_based_similarities(self, song_name, artist_name, songs_data,transformed_matrix):
        return content_recommendation(song_name, artist_name, songs_data,transformed_matrix) # [[0.4, 0.8...]]
        
    def __calculate_collaborative_filtering_similarities(self, song_name, artist_name, songs_data, interaction_matrix):
        return collaborative_recommendation(song_name, artist_name, songs_data, interaction_matrix) # [[0.4, 0.8...]]
    
    def __normalize_similarities(self, similarity_scores):
        minimum = np.min(similarity_scores)
        maximum = np.max(similarity_scores)
        normalized_scores = (similarity_scores - minimum) / (maximum - minimum)
        return normalized_scores
    
    def __weighted_combination(self, content_based_scores, collaborative_filtering_scores):
        weighted_scores = (self.weight_content_based * content_based_scores) + (self.weight_collaborative * collaborative_filtering_scores)
        return weighted_scores
    
    def give_recommendations(self, song_name, artist_name, songs_data, transformed_matrix, interaction_matrix):
        # calculate content based similarities
        content_based_similarities = self.__calculate_content_based_similarities(
            song_name = song_name, 
            artist_name = artist_name, 
            songs_data = songs_data,
            transformed_matrix = transformed_matrix
        )
        
        # calculate collaborative filtering similarities
        collaborative_filtering_similarities = self.__calculate_collaborative_filtering_similarities(
            song_name= song_name,
            artist_name= artist_name,
            songs_data= songs_data, 
            interaction_matrix= interaction_matrix
        )
    
        # normalize content based similarities
        normalized_content_based_similarities = self.__normalize_similarities(content_based_similarities)
        
        # normalize collaborative filtering similarities
        normalized_collaborative_filtering_similarities = self.__normalize_similarities(collaborative_filtering_similarities)
        
        # weighted combination of similarities
        weighted_scores = self.__weighted_combination(
            content_based_scores = normalized_content_based_similarities, 
            collaborative_filtering_scores = normalized_collaborative_filtering_similarities
        )
        
        # index values of recommendations
        idx = np.argsort(weighted_scores.ravel())[-self.number_of_recommendations - 1:][::-1]
        return songs_data.loc[idx, ["name", "artist", "spotify_preview_url"]]

def main():
    # Creating Paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    songs_data_url = data_path / "interim" / "filtered_songs.csv"

    # load the data
    df_songs = pd.read_csv(songs_data_url)
    interaction_matrix = scipy.sparse.load_npz(data_path / "processed" / "interaction_matrix.npz")
    transformed_matrix = scipy.sparse.load_npz(data_path / "processed" / "songs_processed.npz")

    obj = HybridRecommenderSystem(
        number_of_recommendations = 25,
        weight_content_based = 0.8
    )

    recommendations = obj.give_recommendations(
        song_name = "stickwitu",
        artist_name = "the pussycat dolls",
        songs_data = df_songs,
        transformed_matrix = transformed_matrix,
        interaction_matrix = interaction_matrix,
    )

    print(recommendations)

if __name__ == '__main__':
    main()