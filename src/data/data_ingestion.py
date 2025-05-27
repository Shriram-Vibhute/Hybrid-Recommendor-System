import pandas as pd
import pathlib

def load_data(df_songs_path: str, df_users_path: str) -> tuple:
    try:
        df_songs = pd.read_csv(filepath_or_buffer = df_songs_path)
        df_users = pd.read_csv(filepath_or_buffer = df_users_path)
        return df_songs, df_users
    except pd.errors.EmptyDataError as e:
        if str(e).startswith("File " + df_users_path):
            raise ValueError(f"Failed to load data from {df_users_path}. The file is empty.")
        else:
            raise ValueError(f"Failed to load data from {df_songs_path}. The file is empty.")
    except pd.errors.ParserError as e:
        if str(e).startswith("File " + df_users_path):
            raise ValueError(f"Failed to parse data from {df_users_path}. Check the file format.")
        else:
            raise ValueError(f"Failed to parse data from {df_songs_path}. Check the file format.")
    except FileNotFoundError as e:
        if str(e).startswith("File " + df_users_path):
            raise FileNotFoundError(f"Failed to find the file at {df_users_path}")
        else:
            raise FileNotFoundError(f"Failed to find the file at {df_songs_path}")
    except Exception as e:
        if str(e).startswith("File " + df_users_path):
            raise RuntimeError(f"An unexpected error occurred while loading data from {df_users_path}")
        else:
            raise RuntimeError(f"An unexpected error occurred while loading data from {df_songs_path}")

def main() -> None:
    # Creating Paths
    current_path = pathlib.Path(__file__).resolve()
    home_path = current_path.parent.parent.parent
    data_path = home_path / "data"
    data_path.mkdir(parents = True, exist_ok = True)

    # Data Import paths
    songs_data_path = data_path / "raw" / "songs.csv"
    user_data_path = data_path / "raw" / "users.csv"

    # Load the data
    df_songs, df_users = load_data(df_songs_path = songs_data_path, df_users_path = user_data_path)
    
    # Printing the data
    print(df_songs.head())
    print(df_users.head())

if __name__ == "__main__":
    main()