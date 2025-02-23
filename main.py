import db
from pathlib import Path


def main():
    if not Path(db.DB_FILE_PATH).exists():
        db.csv_to_sql(verbose=True)


if __name__ == "__main__":
    main()
