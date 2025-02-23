import pandas as pd
import sqlite3

DB_FILE_PATH = "./database/tfg_db.db"
SQL_FILE_PATH = "./database/tfg_db.sql"


def load_sql_script(conn: sqlite3.Connection):
    """
    Loads a SQL script into a SQLite database.

    Args:
        conn (sqlite3.Connection): The connection to the sqlite database.
    """
    with open(SQL_FILE_PATH, "r") as f:
        script = f.read()
    cursor = conn.cursor()
    cursor.executescript(script)
    conn.commit()


def csv_to_sql(verbose: bool = False):
    """
    Creates a SQLite database from the CSV files in the data/ directory.

    Args:
        verbose (bool, optional): Indicates whether to print progress messages. Defaults to False.
    """
    conn = sqlite3.connect(DB_FILE_PATH)

    if verbose:
        print(f"Creating tables in {DB_FILE_PATH}...")

    load_sql_script(conn)

    if verbose:
        print("Loading datasets...")

    users_en_df = pd.read_csv("data/users_en.csv")
    users_fr_df = pd.read_csv("data/users_fr.csv")
    users_df = pd.concat([users_en_df, users_fr_df])
    users_df.to_sql("users", con=conn, if_exists="replace")

    items_en_df = pd.read_csv("data/items_en.csv")
    items_fr_df = pd.read_csv("data/items_fr.csv")
    items_df = pd.concat([items_en_df, items_fr_df])
    items_df.to_sql("items", con=conn, if_exists="replace")

    explicit_ratings_en_df = pd.read_csv("data/explicit_ratings_en.csv")
    explicit_ratings_fr_df = pd.read_csv("data/explicit_ratings_en.csv")
    explicit_ratings_df = pd.concat([explicit_ratings_en_df, explicit_ratings_fr_df])
    explicit_ratings_df.to_sql("explicit_ratings", con=conn, if_exists="replace")

    implicit_ratings_en_df = pd.read_csv("data/implicit_ratings_en.csv")
    implicit_ratings_fr_df = pd.read_csv("data/implicit_ratings_en.csv")
    implicit_ratings_df = pd.concat([implicit_ratings_en_df, implicit_ratings_fr_df])
    implicit_ratings_df.to_sql("implicit_ratings", con=conn, if_exists="replace")

    if verbose:
        print("Datasets loaded.")

    conn.close()
