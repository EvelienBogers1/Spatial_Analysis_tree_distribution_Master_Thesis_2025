import numpy as np
import pandas as pd
import os

def process_LPJ_GUESS_output(file_path, output_file):
    """
    Processes LPJ-GUESS output files in the specified directory and saves the results to a CSV file.

    Parameters:
        directory (str): Path to the directory containing LPJ-GUESS output files.
        output_file (str): Path to the output CSV file where results will be saved.
    """
    df = pd.read_csv(file_path, delim_whitespace=True)

    filtered_df = df[
        (df['Year'] == 2022) &
        (df['Lon'] == 10.75) &
        (df['Lat'] == 60.25)
    ].copy()

    filtered_df['N'] = filtered_df.groupby('PID').cumcount() + 1

    data = {
        'plot_ID': filtered_df['PID'],
        'N'  : filtered_df['PID'],
        'position' : filtered_df['Pos'],
        'year': filtered_df['Year'],
        'species': filtered_df['PFT'].astype(str),
        'entext': filtered_df['PFT'].astype(str),
        'cmass': filtered_df['Cmass'],
        'crown': filtered_df['CrownA'],
    }

    mapping = {
        "0": "BNE",
        "1": "TeBS",
        "2": "IBS"
    }

    data['entext'] = [mapping.get(val, val) for val in data['entext']]

    data = pd.DataFrame(data)
    data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

# area = 1000

data_name = 'redo_result_4.2'

# input_name = 'Output_0807(1)_vegetation_structure.out'
# input_name = 'Output_before_changes_vegetation_structure9216.out'
input_name = data_name + '_vegetation_structure.out'

# output_name = 'LPJ_processed_output_0807.csv'
output_name = 'LPJ_processed_output_' + data_name + '.csv'

process_LPJ_GUESS_output(input_name, output_name)