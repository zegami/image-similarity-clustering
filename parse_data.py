# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:12:11 2020

@author: dougl
"""

import os
import pandas as pd
from utils import col_alpha_to_num, col_num_to_alpha


def parse_data(data, feature_cols=None, unique_col=None):
    ''' Ensures we're dealing with a pandas DataFrame, and cuts the frame down
    to only the specified data. If no unique_col argument is provided, a new
    column at 'A' will be created called 'ID'.
    
    Args:
        data (pd.DataFrame/str): Data to work with. Can be a DataFrame or a filepath to a .csv.
        
        feature_cols (str/list): Either a list of column strings like ['A', 'B', 'AB'], or 'all'.
        
        unique_col (str): Optional. The index of the column containing unique keys, to be omitted from any mathemetical operations when feature_cols='all' is used.
        
    Returns:
        (pd.DataFrame): The trimmed DataFrame, with the unique column in 'A'.
    '''
    
    if feature_cols is None:
        raise ValueError('Please provide either a list of feature column indices, '\
                         'such as ["B", "C", "AD"], or pass the string "all" to '\
                         'treat every column as numeric data. Pass a unique_col '\
                         'if you want to retain an ID-column, without it affecting '\
                         'feature calculations (usually column "A").')
    
    # Check the input data is a .csv file or DataFrame
    if type(data) == str:
        assert os.path.exists(data),\
            'Parsing data: File not found: "{}"'.format(data)
        assert data.lower().endswith('.csv'),\
            'Parsing data: Requires a .csv file'
        data = pd.read_csv(data, dtype=object)
        
    elif not isinstance(data, pd.DataFrame):
        raise ValueError('Parsing data: "data" arg is not a filepath or pd.DataFrame ({})'\
                         .format(type(data)))
    
    # If passed 'all' for feature_cols, figure out the column indices to use
    if feature_cols == 'all':
        feature_cols = [col_num_to_alpha(i) for i in list(range(len(data.columns)))]
        if unique_col is not None:
            feature_cols.remove(unique_col)
        
    # Shave the DataFrame into only salient data, the unique column forced to 'A'
    column_data_list = []
    
    # Fill the unique 'A' column
    if unique_col is None:
        id_df = pd.DataFrame([str(i) for i in list(range(0, len(data)))], columns=['ID'], dtype=str)
        column_data_list.append(id_df['ID'])
    else:
        n = data.columns[col_alpha_to_num(unique_col)]
        id_df = pd.DataFrame([str(i) for i in list(data[n])], columns=[n], dtype=str)
        column_data_list.append(data[n])
        
    # Fill the numeric columns
    for feat_col in feature_cols:
        idx = col_alpha_to_num(feat_col)
        col_name = data.columns[idx]
        column_data_list.append(data[col_name])
    
    return pd.DataFrame(column_data_list, dtype=object).transpose()