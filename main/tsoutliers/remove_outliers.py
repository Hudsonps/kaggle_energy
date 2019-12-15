'''Eliminates outliers and stitches together the output of R script
that calculates the outlier score
'''

import pathlib
import pandas as pd
import feather
from tqdm import tqdm


if __name__ == '__main__':
    MAIN = pathlib.Path('/Users/palermopenano/personal/kaggle_energy')
    TMP_PATH = MAIN / 'tmp'

    files_list = list(TMP_PATH.glob('outlier_score*'))

    print("Load all...")
    dfs = [feather.read_dataframe(p) for p in tqdm(files_list)]
    df_outliers = pd.concat(dfs, ignore_index=True)

    # !!! Remove timezone information so we can merge it
    # with the meter_readings dataset
    df_outliers.timestamp = \
        df_outliers.timestamp.dt.tz_localize(None)

    # Merge in meter readings
    print("Loading and merging original meter reading data...")
    df_train = pd.read_csv(MAIN / 'data' / 'train.csv')
    df_train.timestamp = pd.to_datetime(
        df_train.timestamp,
        format="%Y-%m-%d %H:%M:%S"
    )
    df_train = df_train.merge(
        df_outliers,
        on=['building_id', 'meter', 'timestamp'],
        how='left')

    # Remove meter readings for which outlier
    # score is greater than 0
    num_out = len(df_train[df_train.outliers > 0.])
    print(
        "",
        f"Number of outliers dropped: {num_out}",
        f"Shape BEFORE dropping: {df_train.shape}",
        sep='\n'
    )
    df_train = df_train[df_train.outliers == 0.]
    print(f"Shape AFTER dropping: {df_train.shape}")
    assert df_train.outliers.isna().sum() == 0

    # Save result
    df_train = df_train.drop('outliers', axis=1)
    feather.write_dataframe(df_train, TMP_PATH / 'train_nooutliers.csv')

    print(df_train.head())
    print(df_train.tail())
