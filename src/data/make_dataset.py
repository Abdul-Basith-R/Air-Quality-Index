# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
import numpy as np
import glob

def combine_state_df(dataset_src, state_name):

    # Read stations_info.csv and drop unnecessary columns
    df_states = pd.read_csv(f'{dataset_src}/stations_info.csv')
    df_states.drop(
        columns=['agency', 'station_location', 'start_month'], inplace=True)
    # Filter df_states based on state_name and sort by city and start_year
    filtered_states = df_states[df_states["state"] ==
                                state_name].sort_values(by=["city", "start_year"])

    # Get state code and state files for the given state_name
    state_code = filtered_states['file_name'].iloc[0][:2]
    state_files = glob.glob(f'{dataset_src}/{state_code}*.csv')

    print(f'Combining a total of {len(state_files)} files...\n')

    combined_df = []
    # print(combined_df)

    for state_file in state_files:
        print(state_file)
        file_name = state_file.split(f'{dataset_src}\\')[1][0:-4]
        print(file_name)
        file_df = pd.read_csv(state_file)
        file_df['city'] = filtered_states[filtered_states['file_name']== file_name]['city'].values[0]
        file_df['city'] = file_df['city'].astype('string')
        combined_df.append(file_df)

    return pd.concat(combined_df)

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    dataset_src = os.path.join("data","raw","air_data")
    print(dataset_src)
    df = combine_state_df(dataset_src, 'Tamil Nadu')

    return df
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    data = main()
    data.to_csv(os.path.join("data","processed","data.csv"))
