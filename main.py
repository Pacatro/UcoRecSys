import db
from pathlib import Path


def main():
    if not Path("./database/tfg_db.db").exists():
        db.csv_to_sql(verbose=True)


if __name__ == "__main__":
    main()
