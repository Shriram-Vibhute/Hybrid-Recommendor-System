"""
Data preprocessing

This script preprocesses the raw data by dropping duplicates, filling
missing values, and converting some columns to lowercase. The cleaned
data is then saved to a new CSV file.

"""

import pandas as pd
import pathlib


def load_data(songs_data_url: str, user_data_url: str) -> tuple:
    """
    Load data from a CSV file.

    Args:
        songs_data_url (str): The path to the songs dataset CSV file.
        user_data_url (str): The path to the users dataset CSV file.

    Returns:
        tuple: A tuple containing the songs dataset and the users dataset.
    """
    try:
        df_songs = pd.read_csv(songs_data_url)
        df_users = pd.read_csv(user_data_url)
        return df_songs, df_users
    except pd.errors.ParserError as e:
        raise
    except Exception as e:
        raise


def clean_song_data(df_songs: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps for songs dataset

    Args:
        df_songs (pd.DataFrame): The songs dataset.

    Returns:
        pd.DataFrame: The cleaned songs dataset.
    """
    try:
        # Drop duplicates
        df_songs = df_songs.drop_duplicates(subset="track_id")

        # Drop columns that are not required
        df_songs = df_songs.drop(columns=["genre", "spotify_id"])

        # Fill missing values
        df_songs = df_songs.fillna({"tags": "no_tags"})

        # Convert some columns to lowercase
        df_songs = df_songs.assign(
            name=lambda x: x["name"].str.lower(),
            artist=lambda x: x["artist"].str.lower(),
            tags=lambda x: x["tags"].str.lower(),
        ).reset_index(drop=True)

        return df_songs
    except pd.errors.ParserError as e:
        raise
    except Exception as e:
        raise


def clean_users_data(df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps for users dataset

    Args:
        df_users (pd.DataFrame): The users dataset.

    Returns:
        pd.DataFrame: The cleaned users dataset.
    """
    return df_users


def save_data(df_songs: pd.DataFrame, df_users: pd.DataFrame, save_path: pathlib.Path) -> None:
    """
    Save the cleaned data to a new CSV file.

    Args:
        df_songs (pd.DataFrame): The cleaned songs dataset.
        df_users (pd.DataFrame): The cleaned users dataset.
        save_path (pathlib.Path): The path to save the cleaned data.
    """
    try:
        # Create the save path if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the data
        save_path_songs = save_path / "songs_interim.csv"
        save_path_users = save_path / "users_interim.csv"
        df_songs.to_csv(save_path_songs, index=False)
        df_users.to_csv(save_path_users, index=False)
    except Exception as e:
        raise


def main() -> None:
    """
    The main function.
    """
    # Create paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)

    songs_data_url = data_path / "raw" / "songs.csv"
    user_data_url = data_path / "raw" / "users.csv"

    save_path = data_path / "interim"

    # Load the data
    df_songs, df_users = load_data(songs_data_url, user_data_url)

    # Clean the data
    df_songs = clean_song_data(df_songs)
    df_users = clean_users_data(df_users)

    # Save the data
    save_data(df_songs=df_songs, df_users=df_users, save_path=save_path)


if __name__ == "__main__":
    main()