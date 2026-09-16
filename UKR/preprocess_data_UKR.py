import numpy as np
import pandas as pd

path_to_data = './UKR/01_treedata-biomass_TMt_UKR.csv'
data = pd.read_csv(path_to_data)

column_map = {
    'tree.id': 'N',
    'tree.coord.x': 'x',
    'tree.coord.y': 'y',
    'census.date': 'year',
    'species.cor': 'species',
}

output_df = (data
            .rename(columns=column_map)
            .assign(entext=data['species.cor'])  # Create a new 'entext' column from 'species.cor'
            [['N', 'x', 'y', 'year', 'species', 'entext']])

output_df = output_df.reset_index(drop=True)
output_df.to_csv('UKR_data_clean.csv', index=False)
