from tarfile import data_filter
import numpy as np
import pandas as pd
import dask
import pathlib
import dask.dataframe as dd
import scipy
from scipy.sparse import csr_matrix, save_npz

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
    
def interaction_matrix(dd_users: dd) -> scipy.sparse.csr_matrix:
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
        return sparse_matrix # dd_users['track_id'].cat.categories.unique()
    except TypeError as e:
        raise TypeError(f"Failed to build interaction matrix. The argument dd_users should be a dask dataframe. {e}")
    except ValueError as e:
        raise ValueError(f"Failed to build interaction matrix. The argument dd_users should be a dask dataframe with user_id, track_id and playcount columns. {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while building interaction matrix. {e}")
    
def save_data(interaction_matrix: scipy.sparse.csr_matrix, save_path: str) -> None:
    """Saving all the Atrifacts"""
    try:
        # Save interaction matrix
        save_matrix_path = save_path / "interaction_matrix"
        save_npz(save_matrix_path, interaction_matrix)
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
        dd_users = dd.read_csv(urlpath = users_data_url)

        # Transform data
        dd_users = transform_data(dd_users = dd_users)

        # Interaction Matrix
        interaction_array = interaction_matrix(dd_users = dd_users)

        # Saving the transformed data
        save_data(interaction_matrix = interaction_array, save_path = save_data_path)
    except Exception as e:
        raise RuntimeError("Unexpected error occured!")

if __name__ == '__main__':
    main()