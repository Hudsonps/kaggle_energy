'''Creates a small dataset for testing code'''

import pathlib
import pandas as pd
import utils


if __name__ == '__main__':
    MAIN = pathlib.Path('/Users/palermopenano/personal/kaggle_energy')

    # Load sample data
    df = pd.read_csv(MAIN / 'data' / 'train_50buildings.csv')

    # Create sample data
    # df = pd.read_csv(MAIN / 'data' / 'train.csv')
    # df, _ = utils.df_sample_random_buildings(df, 'building_id', n=50)
    # df.to_csv(MAIN / 'data' / 'train_50buildings.csv', index=False)

    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")



    print(df.shape)
    all_zero_meter_zero_cond = (df.meter == 0) & (df.meter_reading == 0.)
    df = df.loc[~all_zero_meter_zero_cond, :]
    df = df.sort_values('timestamp')
    df.to_csv(MAIN / 'data' / 'meters.csv', index=False)
    print(df.shape)
