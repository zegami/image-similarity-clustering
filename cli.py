# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:17:54 2020

@author: Doug Lawrence
"""

import argparse
import sys
import string
import pandas as pd


def col_num_to_alpha(n, zero_indexed=True):
    ''' colnum_string(28) = 'AB' '''
    if zero_indexed:
        n = n + 1
    string = ''
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        string = chr(65 + remainder) + string
    return string


def col_alpha_to_num(col, zero_indexed=True):
    num = 0
    for c in col:
        if c in string.ascii_letters:
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
    return num - 1 if zero_indexed else num


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='features | tsne | umap')
    parser.add_argument('--source', '-s', help='(Use this OR --data) The source csv file to read data from.')
    parser.add_argument('--data', '-d', help='(Use this OR --source) A pandas dataframe to read data from.')
    parser.add_argument('--numeric-cols', '-n', help='Numeric data column indices to perform math on. Ex: "B,C,F", use "all" to consider all columns (excluding optional index column).')
    parser.add_argument('--index-col', '-i', help='The column index containing unique IDs for each row. Not required.')
    parser.add_argument('--reduce', '-r', help='How many dimensions to reduce features to. Default is 2.', default='2')
    
    args = parser.parse_args(argv[1:])
    
    
    def parse_data():
        ''' Ensures we're actually dealing with a pandas DataFrame, and cuts
        the frame down to only the specified data. Also returns whether or not
        the FIRST column in this reduced frame contains unique IDs, rather than
        numeric values. '''        
        
        def parse_numeric_cols(data, string, index_col):
            ''' Returns the columns indices associated with numeric data. '''
            if string == 'all':
                col_indices = [col_num_to_alpha(i) for i in list(range(len(data.columns)))]
                if index_col is not None:
                    col_indices.remove(index_col)
                return col_indices
            
            col_indices = [part for part in string.split(',') if part.strip() != '']
            return col_indices
        
        
        def shave_df(data, numeric_cols, index_col):
            ''' Returns a cut-down version of the input data, and whether or not the
            first column is an index column or not. '''
            
            column_data_list = []
            if index_col is not None:
                column_data_list.append(data[data.columns[col_alpha_to_num(index_col)]])
            for num_col in numeric_cols:
                column_data_list.append(data[data.columns[col_alpha_to_num(num_col)]])
            
            return pd.DataFrame(column_data_list).transpose()
            
        
        # Make sure we've actually been given data
        if args.source is None and args.data is None:
            raise Exception('No data provided. Pass in a csv file with --source, or a dataframe with --data')
            
        # Make sure we're explicitly only using one source for data
        if args.source is not None and args.data is not None:
            raise Exception('Please provide --source OR --data, not both')
        
        # Read the data either directly or from a file
        if args.data is not None:
            assert isinstance(args.data, pd.DataFrame), '--data value is not a pandas DataFrame'
            data = args.data
        else:
            data = pd.read_csv(args.source)
            
        # Make sure we know what columns to use
        if args.numeric_cols is None:
            raise Exception('No data column indices provided. Example usage: "--numeric-cols B,C,F", "--numeric-cols all"')
        numeric_cols = parse_numeric_cols(data, args.numeric_cols, args.index_col)
        
        return [shave_df(data, numeric_cols, args.index_col), args.index_col is not None]
            
    
    # Feature extraction
    if args.mode == 'features':
        df, has_index_col = parse_data()
        print(df)
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))