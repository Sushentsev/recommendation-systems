import os

import pandas as pd

from hw2.load_data import load_members


def handle_city(user_features_df: pd.DataFrame):
    pass


def handle_bd(user_features_df: pd.DataFrame):
    def bd_category(value: int) -> str:
        if value == 0 or value > 100:
            return "<UNK>"
        elif 0 < value <= 16:
            return "child"
        elif 16 < value <= 30:
            return "young"
        elif 30 < value <= 45:
            return "middle_age"
        else:
            return "old_age"

    user_features_df["bd_category"] = user_features_df.bd.apply(bd_category).astype("category")
    user_features_df.drop(columns="bd", inplace=True)


def handle_gender(user_features_df: pd.DataFrame):
    user_features_df.gender = user_features_df.gender.cat.add_categories("<UNK>").fillna(value="<UNK>")


def handle_registration_init_time(user_features_df: pd.DataFrame):
    user_features_df["registration_init_year"] = user_features_df.registration_init_time\
        .apply(lambda x: x.year).astype("int")
    user_features_df.drop(columns="registration_init_time", inplace=True)


def handle_expiration_date(user_features_df: pd.DataFrame):
    user_features_df["expiration_date_year"] = user_features_df.expiration_date\
        .apply(lambda x: x.year).astype("int")
    user_features_df.drop(columns="expiration_date", inplace=True)


def create_user_features(data_dir: str):
    users_df = load_members(os.path.join(data_dir, "members.csv"))
    users_features_df = users_df

    handle_city(users_features_df)
    handle_bd(users_features_df)
    handle_gender(users_features_df)
    handle_registration_init_time(users_features_df)
    handle_expiration_date(users_features_df)

    return users_features_df
