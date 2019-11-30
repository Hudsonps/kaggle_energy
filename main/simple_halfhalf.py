'''
Ported from https://www.kaggle.com/rohanrao/ashrae-half-and-half
'''

import gc
import os
import random
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from utils import *


def prepare_data(X, building_data, weather_data, test=False):
    """
    Preparing final dataset with all features.
    """

    X = X.merge(building_data, on="building_id", how="left")
    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")

    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    X.square_feet = np.log1p(X.square_feet)

    if not test:
        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)

    gc.collect()

    holidays = [
        "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
        "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
        "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
        "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
        "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
        "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
        "2019-01-01"
    ]

    X["hour"] = X.timestamp.dt.hour
    X["weekday"] = X.timestamp.dt.weekday
    X["is_holiday"] = (
        X.timestamp.dt.date.
        astype("str").
        isin(holidays)
    ).astype(int)

    drop_features = [
        "timestamp", "sea_level_pressure", "wind_direction", "wind_speed"
    ]

    X.drop(drop_features, axis=1, inplace=True)

    if test:
        row_ids = X.row_id
        X.drop("row_id", axis=1, inplace=True)
        return X, row_ids
    else:
        y = np.log1p(X.meter_reading)
        X.drop("meter_reading", axis=1, inplace=True)
        return X, y


if __name__ == '__main__':
    MAIN = pathlib.Path('/Users/palermopenano/personal/kaggle_energy')
    SUBMISSIONS_PATH = MAIN / 'submissions'

    ##############
    # Parameters #
    ##############
    sample = True
    submission_name = "submission_2019-11-30_simple_halfhalf.csv"

    myfavouritenumber = 0
    seed = myfavouritenumber
    random.seed(seed)

    #############
    # Load Data #
    #############
    print("Loading data...")
    df_train = pd.read_csv(MAIN / 'data' / 'train.csv')
    building = pd.read_csv(MAIN / 'data' / 'building_metadata.csv')
    le = LabelEncoder()
    building.primary_use = le.fit_transform(building.primary_use)
    weather_train = pd.read_csv(MAIN / 'data' / 'weather_train.csv')

    # Take only a random sample of n buildings
    randbuilding = None
    if sample:
        print("Taking a random sample of buildings...")
        df_train, randbuilding = \
            df_sample_random_buildings(df_train, 'building_id', n=5)
        print(randbuilding)
    print(df_train.shape)

    #######################
    # Reduce Memory Usage #
    #######################
    print("Reducing memory usage...")
    df_train = reduce_mem_usage(df_train, use_float16=True)
    building = reduce_mem_usage(building, use_float16=True)
    weather_train = reduce_mem_usage(weather_train, use_float16=True)

    #########################
    # Prepare Training Data #
    #########################
    X_train, y_train = prepare_data(df_train, building, weather_train)
    del df_train, weather_train
    gc.collect()

    #####################
    # Two-Fold LightGBM #
    #####################
    print("\n==================\n")
    X_half_1 = X_train[:int(X_train.shape[0] / 2)]
    X_half_2 = X_train[int(X_train.shape[0] / 2):]

    y_half_1 = y_train[:int(X_train.shape[0] / 2)]
    y_half_2 = y_train[int(X_train.shape[0] / 2):]

    categorical_features = [
        "building_id", "site_id",
        "meter", "primary_use",
        "hour", "weekday"
    ]

    d_half_1 = lgb.Dataset(
        X_half_1,
        label=y_half_1,
        categorical_feature=categorical_features,
        free_raw_data=False
    )
    d_half_2 = lgb.Dataset(
        X_half_2,
        label=y_half_2,
        categorical_feature=categorical_features,
        free_raw_data=False
    )

    watchlist_1 = [d_half_1, d_half_2]
    watchlist_2 = [d_half_2, d_half_1]

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 40,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse"
    }

    print("Building model with first half and validating on second half:")
    model_half_1 = lgb.train(
        params,
        train_set=d_half_1,
        num_boost_round=1000,
        valid_sets=watchlist_1,
        verbose_eval=200,
        early_stopping_rounds=200
    )

    print("Building model with second half and validating on first half:")
    model_half_2 = lgb.train(
        params,
        train_set=d_half_2,
        num_boost_round=1000,
        valid_sets=watchlist_2,
        verbose_eval=200,
        early_stopping_rounds=200
    )

    #####################
    # Prepare Test Data #
    #####################
    print(
        "\n==================\n",
        "Loading test set...",
        sep='\n'
    )
    df_test = pd.read_csv(MAIN / 'data' / 'test.csv')
    weather_test = pd.read_csv(MAIN / 'data' / 'weather_test.csv')

    df_test = reduce_mem_usage(df_test)
    weather_test = reduce_mem_usage(weather_test)

    if sample:
        df_test = df_test[df_test['building_id'].isin(randbuilding)]
        print("Shape of test data: ", df_test.shape)

    X_test, row_ids = prepare_data(df_test, building, weather_test, test=True)

    ######################
    # Prepare Submission #
    ######################
    print(
        "\n==================\n",
        "Generating predicitons on test set...",
        sep='\n'
    )
    raw_pred_1 = model_half_1.predict(
        X_test,
        num_iteration=model_half_1.best_iteration
    )
    pred = np.expm1(raw_pred_1) / 2
    del model_half_1
    gc.collect()

    raw_pred_2 = model_half_2.predict(
        X_test,
        num_iteration=model_half_2.best_iteration
    )
    pred += np.expm1(raw_pred_2) / 2
    del model_half_2
    gc.collect()

    print("Saving predictions as csv...")
    submission = pd.DataFrame(
        {"row_id": row_ids, "meter_reading": np.clip(pred, 0, a_max=None)}
    )
    submission.to_csv(
        SUBMISSIONS_PATH / (submission_name + '.csv'), index=False
    )
