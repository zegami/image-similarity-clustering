# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:32:06 2020

@author: dougl
"""

from umap import UMAP
import pandas as pd


def umap(features, dims=2, write_to=None):
    ''' Reduces the features in the parsed pd.DataFrame 'features' into 'dims'
    dimensions (default 2). Writes the output to 'write_to' if provided, in
    .csv format. Returns the feature DataFrame.
    '''
    
    if dims != 2:
        print('UMAP: Not currently supporting anything but 2-dim reduction')
    
    id_col_name = features.columns[0]
        
    print('UMAP: Reducing features to 2 dimensions'.format(dims))
    
    # Don't consider the first unique ID column
    features_salient = features.copy().drop(columns=[id_col_name], axis=1)
    
    reduced = pd.DataFrame(UMAP().fit_transform(features_salient))
    reduced.insert(0, id_col_name, features[[id_col_name]])
    
    print('Success')
    
    if write_to is not None:
        try:
            reduced.to_csv(write_to, index=False)
            print('Wrote reduced features to "{}"'.format(write_to))
        except Exception as e:
            print('\nWARNING - Could not write results to file: "{}"'.format(e))
    
    return reduced
