DROP TABLE IF EXISTS explicit_ratings;
DROP TABLE IF EXISTS implicit_ratings;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS items;

CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY NOT NULL,
    job TEXT
);

CREATE TABLE IF NOT EXISTS items (
    item_id INTEGER PRIMARY KEY NOT NULL,
    language TEXT,
    name TEXT,
    nb_views INTEGER,
    description TEXT,
    created_at INTEGER,
    difficulty TEXT,
    job TEXT,
    software TEXT,
    theme TEXT,
    duration REAL,
    type TEXT
);

CREATE TABLE IF NOT EXISTS explicit_ratings (
    user_id INTEGER NOT NULL,
    item_id INTEGER NOT NULL,
    watch_percentage REAL,
    created_at TEXT,  -- Almacenar fechas como TEXT en formato ISO (YYYY-MM-DD)
    rating INTEGER,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (item_id) REFERENCES items (item_id)
);

CREATE TABLE IF NOT EXISTS implicit_ratings (
    item_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    created_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (item_id) REFERENCES items (item_id)
);

-- SELECT * FROM users;
-- SELECT * FROM items;
-- SELECT * FROM explicit_ratings;
-- SELECT * FROM implicit_ratings;
--
-- DELETE FROM users;
-- DELETE FROM items;
-- DELETE FROM explicit_ratings;
-- DELETE FROM implicit_ratings;
