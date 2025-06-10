import pandas as pd
from typing import Literal


def load_mars() -> pd.DataFrame:
    explicit_df_en = pd.read_csv("./data/mars_dataset/explicit_ratings_en.csv")
    explicit_df_fr = pd.read_csv("./data/mars_dataset/explicit_ratings_fr.csv")

    items_en = pd.read_csv("./data/mars_dataset/items_en.csv")
    items_fr = pd.read_csv("./data/mars_dataset/items_fr.csv")

    df_explicit = pd.concat([explicit_df_en, explicit_df_fr], ignore_index=True)
    df_items = pd.concat([items_en, items_fr], ignore_index=True)

    df_explicit["created_at"] = pd.to_datetime(df_explicit["created_at"])
    df_items = df_items.drop(columns=["created_at"])

    df = pd.merge(df_explicit, df_items, on="item_id", how="inner")

    df.rename(
        columns={"Difficulty": "difficulty", "type": "item_type"},
        inplace=True,
    )

    features = [
        "user_id",
        "item_id",
        "item_type",
        "difficulty",
        "nb_views",
        "watch_percentage",
        "rating",
    ]

    return df[features]


def load_itm() -> pd.DataFrame:
    ratings_df = pd.read_csv("./data/itm_dataset/ratings.csv")
    items_df = pd.read_csv("./data/itm_dataset/items.csv")
    users_df = pd.read_csv("./data/itm_dataset/users.csv")

    merged_df = pd.merge(left=items_df, right=ratings_df, how="inner", on="Item")
    merged_df = pd.merge(left=merged_df, right=users_df, how="inner", on="UserID")
    merged_df = merged_df.rename(
        columns={"UserID": "user_id", "Item": "item_id", "Rating": "rating"}
    )

    features = [
        "user_id",
        "item_id",
        "Title",
        "Semester",
        "Class",
        "App",
        "Lockdown",
        "Ease",
        " Age",
        "Married",
        "rating",
    ]

    return merged_df[features]


# def load_coursera() -> pd.DataFrame:
#     # df = pd.read_csv("./data/coursera_dataset/Coursera.csv")
#     # num_users = 2000
#     # num_interactions = 20000
#     # user_ids = np.random.randint(1, num_users + 1, size=num_interactions)
#     # course_titles = np.random.choice(df["Course Name"], size=num_interactions)
#     # interaction_types = np.random.choice(
#     #     ["view", "enroll", "complete", "rate"],
#     #     size=num_interactions,
#     #     p=[0.4, 0.3, 0.2, 0.1],
#     # )
#     #
#     # # Asegurar que todos tengan un rating (aunque no sea realista para algunas interacciones)
#     # ratings = np.random.uniform(1, 5, size=num_interactions)
#     #
#     # interactions = pd.DataFrame(
#     #     {
#     #         "user_id": user_ids,
#     #         "Course Name": course_titles,
#     #         "interaction_type": interaction_types,
#     #         "rating": ratings,
#     #     }
#     # )
#     #
#     # df_merged = interactions.merge(df, on="Course Name", how="left", indicator=True)
#     # df_merged = df_merged[df_merged["_merge"] == "both"].drop(columns=["_merge"])
#     #
#     # df_merged = df_merged.rename(columns={"Course Name": "item_id"})
#
#     df = pd.read_csv("./data/coursera_dataset/Coursera_reviews.csv")
#
#     df.rename(columns={"reviewers": "user_id", "course_id": "item_id"}, inplace=True)
#
#     features = [
#         "user_id",
#         "item_id",
#         # "Difficulty Level",
#         # "University",
#         "rating",
#     ]
#
#     return df[features].head(80000)
#


def load_data(
    dataset_name: Literal["mars", "itm"],
) -> pd.DataFrame:
    match dataset_name:
        case "mars":
            return load_mars()
        case "itm":
            return load_itm()
        # case "coursera":
        #     return load_coursera()
