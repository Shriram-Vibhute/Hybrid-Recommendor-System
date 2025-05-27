import pandas as pd
import pathlib
import scipy
from scipy.sparse import save_npz

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

def train_transformer(df_songs: pd.DataFrame) -> ColumnTransformer:
    """Training the Transformer"""
    try:
        # Columns to Transform
        frequency_enode_cols = ['year'] # Due to high cardinality and preserve the current trend
        ohe_cols = ['artist', "time_signature", "key"]
        tfidf_col = 'tags'
        standard_scale_cols = ["loudness"]
        min_max_scale_cols = ["duration_ms", "danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

        # Building the Transformer 
        transformer = ColumnTransformer(
            transformers=[
                ("frequency_encode", CountEncoder(normalize=True,return_df=True), frequency_enode_cols),
                ("ohe", OneHotEncoder(handle_unknown="ignore"), ohe_cols),
                ("tfidf", TfidfVectorizer(max_features=85), tfidf_col),
                ("standard_scale", StandardScaler(), standard_scale_cols),
                ("min_max_scale", MinMaxScaler(), min_max_scale_cols)
            ], 
            remainder='passthrough', n_jobs=-1, force_int_remainder_cols = False
        )

        # Fit the transformer
        transformer.fit(df_songs)

        # Return the Transformer
        return transformer
    except KeyError as e:
        raise KeyError(f"The column {e} does not exist in the DataFrame")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while training the transformer. {e}")
    
def transform_data(df_songs: pd.DataFrame, transformer: ColumnTransformer) -> scipy.sparse.csr_matrix:
    """Transforming the DataFrame"""
    try:
        # Transform the data
        transformed_data = transformer.transform(df_songs)
        return transformed_data
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while transforming the data. {e}")

def save_transformed_data(transformed_df: scipy.sparse.csr_matrix, save_path: str) -> None:
    """Saving all the Atrifacts"""
    try:
        # Save the transformed data in sparse format
        save_npz(save_path, transformed_df)
    except OSError as e:
        raise OSError(f"Failed to save the transformed data to {save_path}. {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while saving the transformed data. {e}")

def main():
    try:
        # Creating Paths
        current_path = pathlib.Path(__file__).resolve()
        home_path = current_path.parent.parent.parent
        data_path = home_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        songs_data_url = data_path / "interim" / "filtered_songs.csv"

        # Storing Transformed Data
        save_data_path = data_path / "processed" / "songs_processed"

        # Loading Data
        df_songs = pd.read_csv(filepath_or_buffer = songs_data_url)
        df_songs = df_songs.drop(columns = ['track_id', 'name', 'spotify_preview_url'])
        df_songs['year'] = df_songs['year'].astype(str)  # or .astype('category')

        # Training Transformer
        transformer = train_transformer(df_songs = df_songs)

        # Data Transformation
        transformed_df = transform_data(df_songs = df_songs, transformer = transformer)

        # Saving the transformed data
        save_transformed_data(transformed_df = transformed_df, save_path = save_data_path)
    except Exception as e:
        raise RuntimeError("Unexpected error occured!")

if __name__ == '__main__':
    main()