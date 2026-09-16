import numpy as np
import pandas as pd

path_to_data = 'trees_suserup.csv'

data = pd.read_csv(path_to_data)

column_map = {
    'indiv_nr': 'N',
    'x_utmetrs89': 'x',
    'y_utmetrs89': 'y',
    'aarstal': 'year',
    'treespecies': 'species',
    'entext': 'entext'
}

output_df = (data
            .rename(columns=column_map)
            [['N', 'x', 'y', 'year', 'species', 'entext']])
output_df = output_df.reset_index(drop=True)

output_df.to_csv('suserup_data_clean.csv', index=False)