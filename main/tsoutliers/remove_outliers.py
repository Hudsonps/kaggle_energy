'''Eliminates and stitches together the output of R script
that calculates the outlier score
'''

import pathlib
import pandas as pd
import feather
from tqdm import tqdm

if __name__ == '__main__':
    TMP_PATH = pathlib.Path('/Users/palermopenano/personal/kaggle_energy/tmp')

    files_list = list(TMP_PATH.glob('outlier_score*'))
    main_df = feather.read_dataframe(files_list[0])

    for o in tqdm(files_list[1:100]):
    # for o in tqdm(files_list):
        df = feather.read_dataframe(o)
        main_df = pd.concat([df, main_df], ignore_index=True)
        # print(
        #     o.name,
        #     main_df.shape,
        #     sep='\n'
        # )

    feather.write_dataframe(main_df, TMP_PATH / 'train_nooutliers.csv')

    print(main_df.head())
    print(main_df.tail())
