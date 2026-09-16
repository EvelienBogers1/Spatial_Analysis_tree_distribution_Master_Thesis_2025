import numpy as np
import pandas as pd

data_sier = pd.read_csv('projected_points_sierpinski_uniform_random.csv')
data_moore = pd.read_csv('projected_points_moore_uniform_random.csv')

data_sier['entext'] = 'Sierpinski'
data_moore['entext'] = 'Moore'

output_df = pd.concat([data_sier, data_moore], ignore_index=True)
output_df = output_df.reset_index(drop=True)

output_df.to_csv('uniform_data_clean.csv', index=False)
print("Combined data saved to 'uniform_data_clean.csv'")