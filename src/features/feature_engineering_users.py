import numpy as np
import pandas as pd
import dask
import pathlib
import dask.dataframe as dd
import scipy
from scipy.sparse import csr_matrix, save_npz

def load_data(songs_data_url: str, users_data_url: str) -> tuple:
    """Loading Data"""
    try:
        # In pandas format
        df_songs = pd.read_csv(songs_data_url)
        df_users = pd.read_csv(users_data_url)

        # In dask dataframe format
        dd_users = dd.read_csv(users_data_url)
        return df_songs, df_users, dd_users
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Failed to load data from {users_data_url}. The file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse data from {users_data_url}. Check the file format.")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to find the file at {users_data_url}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading data from {users_data_url}")

def filter_unique_songs(df_songs: pd.DataFrame, df_users: pd.DataFrame) -> pd.DataFrame:
    try:
        # Finding unique song ids from users dataset
        unique_track_ids = df_users.loc[:,"track_id"].unique()
        unique_track_ids = unique_track_ids.tolist()

        # Finding all the unique songs which is present in both the dataset
        filtered_songs = df_songs[df_songs["track_id"].isin(unique_track_ids)]
        filtered_songs = filtered_songs.reset_index(drop=True)
        return filtered_songs
    except KeyError as e:
        raise KeyError(f"The column {e} is not present in the dataset. Please check the dataset.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while filtering the dataset. {e}")

def transform_data(dd_users: dd) -> dask.dataframe:
    try:
        # Ensure playcount is numeric
        dd_users['playcount'] = dd_users['playcount'].astype(np.float64)

        # Changing the astype of user_id and track_id to "dask category" using categorize function
        dd_users = dd_users.categorize(columns=['user_id', 'track_id']) # Basically, this function is similar to LabelEncoder in sklearn

        # Convert user_id and track_id to numeric indices - This is necessary for creating a sparse matrix later
        user_mapping = dd_users['user_id'].cat.codes 
        track_mapping = dd_users['track_id'].cat.codes
        # .cat is an accessor just like .str, .codes is encoding into integers

        # Testing the difference
        """
            print(dd_users['track_id'].cat.categories) # This is in the sorted form
            print(dd_users.compute()['track_id']) # This is not in sorted form
        """

        # Concatinating those encoded columns into our primary dataset
        dd_users = dd_users.assign(
            user_idx=user_mapping,
            track_idx=track_mapping
        )
        return dd_users
    except KeyError as e:
        raise KeyError(f"The column {e} is not present in the dataset. Please check the dataset.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while transforming the dataset. {e}")
    
def interaction_matrix(dd_users) -> scipy.sparse.csr_matrix:
    # Dask doesn't support pivot tables directly, so we have to build interaction matrix manually
    # Why Dask - Because Pivot Table requires 60GBs of memory to store it
    try:
        # Aggrigating
        interaction_array = dd_users.groupby(['track_idx', 'user_idx'])['playcount'].sum().reset_index()
        # Retransforming into pandas dataframe
        interaction_array = interaction_array.compute()

        # Fetching all the columns seperately
        row_indices = interaction_array['track_idx']
        col_indices = interaction_array['user_idx']
        values = interaction_array['playcount']

        # Build a sparse matrix
        n_tracks = row_indices.nunique()
        n_users = col_indices.nunique()
        sparse_matrix = csr_matrix((values, (row_indices, col_indices)), shape=(n_tracks, n_users))
        return sparse_matrix, dd_users['track_id'].cat.categories.unique()
    except TypeError as e:
        raise TypeError(f"Failed to build interaction matrix. The argument dd_users should be a dask dataframe. {e}")
    except ValueError as e:
        raise ValueError(f"Failed to build interaction matrix. The argument dd_users should be a dask dataframe with user_id, track_id and playcount columns. {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while building interaction matrix. {e}")
    
def save_data(filtered_songs: pd.DataFrame, interaction_matrix: scipy.sparse.csr_matrix, unique_track_ids: pd.Series, save_path: str) -> None:
    """Saving all the Atrifacts"""
    try:
        # Saving Filtered Songs data
        save_songs_path = save_path / "filtered_songs.csv"
        filtered_songs.sort_values(by = 'track_id').to_csv(save_songs_path)

        # Save interaction matrix
        save_matrix_path = save_path / "interaction_matrix"
        save_npz(save_matrix_path, interaction_matrix)

        # Save unique track ids
        unique_track_ids_path = save_path / "unique_track_ids"
        np.save(unique_track_ids_path, unique_track_ids, allow_pickle = True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to save the transformed data. The specified directory {save_path} does not exist. {e}")
    except NotADirectoryError as e:
        raise NotADirectoryError(f"Failed to save the transformed data. The specified path {save_path} is not a directory. {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while saving the transformed data. {e}")

def main():
    try:
        # Creating Paths
        current_path = pathlib.Path(__file__).resolve()
        home_path = current_path.parent.parent.parent
        data_path = home_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        songs_data_url = data_path / "interim" / "songs_interim.csv"
        users_data_url = data_path / "interim" / "users_interim.csv"

        # Storing Transformed Data
        save_data_path = data_path / "processed"

        # Loading Data
        df_songs, df_users, dd_users = load_data(songs_data_url = songs_data_url, users_data_url = users_data_url)

        # Filter unique songs
        filtered_songs = filter_unique_songs(df_songs = df_songs, df_users = df_users)

        # Transform data
        dd_users = transform_data(dd_users = dd_users)

        # Interaction Matrix
        interaction_array, unique_track_ids = interaction_matrix(dd_users = dd_users)

        # Saving the transformed data
        save_data(filtered_songs = filtered_songs, interaction_matrix = interaction_array, unique_track_ids = unique_track_ids, save_path = save_data_path)
    except Exception as e:
        raise RuntimeError("Unexpected error occured!")

if __name__ == '__main__':
    main()