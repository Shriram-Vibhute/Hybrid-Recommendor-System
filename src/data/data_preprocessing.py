import pandas as pd
import pathlib
from data_ingestion import load_data

def preprocess_song_data(df_songs: pd.DataFrame) -> pd.DataFrame:
    try:
        # Drop duplicates
        df_songs = df_songs.drop_duplicates(subset="track_id")

        # Drop columns that are not required
        df_songs = df_songs.drop(columns=["genre", "spotify_id"])

        # Fill missing values
        df_songs = df_songs.fillna({"tags": "no_tags"})

        # Convert some columns to lowercase
        df_songs = df_songs.assign(
            name = lambda x: x["name"].str.lower(),
            artist = lambda x: x["artist"].str.lower(),
            tags = lambda x: x["tags"].str.lower(),
        ).reset_index(drop=True)

        # Returning the data
        return df_songs

    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Failed to load data. The file is empty.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading data ")

def preprocess_user_data(df_users: pd.DataFrame) -> pd.DataFrame:
    # No preprocessing require - returning the dataframe as it is
    return df_users

def filter_unique_songs(df_songs: pd.DataFrame, df_users: pd.DataFrame) -> pd.DataFrame:
    try:
        # Finding unique song ids from users dataset
        unique_track_ids = df_users.loc[:,"track_id"].unique()
        unique_track_ids = unique_track_ids.tolist()

        # Finding all the unique songs which is present in both the dataset
        filtered_songs = df_songs[df_songs["track_id"].isin(unique_track_ids)]
        filtered_songs = filtered_songs.reset_index(drop=True)
        return filtered_songs.sort_values(by = 'track_id')
    except KeyError as e:
        raise KeyError(f"The column {e} is not present in the dataset. Please check the dataset.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while filtering the dataset. {e}")


def save_data(df_songs: pd.DataFrame, df_users: pd.DataFrame, filtered_songs: pd.DataFrame,  songs_save_path: str, users_save_path: str, filtered_songs_save_path: str) -> None:
    # Saving the data
    try:
        df_songs.to_csv(songs_save_path, index = False)
        df_users.to_csv(users_save_path, index = False)
        filtered_songs.to_csv(filtered_songs_save_path, index = False)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to save the data. Check the file path {songs_save_path} and {users_save_path}.") from e
    except Exception as e:
        raise RuntimeError("An unexpected error occured")

def main() -> None:
    # Create paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    songs_data_url = data_path / "raw" / "songs.csv"
    user_data_url = data_path / "raw" / "users.csv"

    save_path = data_path / "interim"
    save_path.mkdir(parents=True, exist_ok=True)

    songs_save_path = save_path / "songs_interim.csv"
    users_save_path = save_path / "users_interim.csv"
    filtered_songs_save_path = save_path / "filtered_songs.csv" 

    # Load the data
    df_songs, df_users = load_data(df_songs_path = songs_data_url, df_users_path = user_data_url)

    # Clean the data
    df_songs = preprocess_song_data(df_songs)
    df_users = preprocess_user_data(df_users)

    # Filtering unique songs
    filtered_songs = filter_unique_songs(df_songs = df_songs, df_users = df_users)

    # Save the data
    save_data(df_songs = df_songs, df_users = df_users, filtered_songs = filtered_songs, songs_save_path = songs_save_path, users_save_path = users_save_path, filtered_songs_save_path = filtered_songs_save_path)

if __name__ == "__main__":
    main()