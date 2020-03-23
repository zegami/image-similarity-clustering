# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:55:14 2020

@author: dougl
"""

from sklearn.manifold import TSNE
import pandas as pd
        

def tsne(features, dims=2, write_to=None, tsne_kwargs={}):
    ''' Reduces the features in the parsed pd.DataFrame 'features' into 'dims'
    dimensions (default 2). Writes the output to 'write_to' if provided, in
    .csv format. Returns the feature DataFrame.
    '''
    
    id_col_name = features.columns[0]
    tsne_kwargs['n_components'] = dims
        
    print('t-SNE: Reducing features to {} dimensions'.format(dims))
    
    # Don't consider the first unique ID column
    features_salient = features.copy().drop(columns=[id_col_name], axis=1)
    
    reduced = pd.DataFrame(TSNE(**tsne_kwargs).fit_transform(features_salient))
    reduced.insert(0, id_col_name, features[[id_col_name]])
    
    print('Success')
    
    if write_to is not None:
        try:
            reduced.to_csv(write_to, index=False)
            print('Wrote reduced features to "{}"'.format(write_to))
        except Exception as e:
            print('\nWARNING - Could not write results to file: "{}"'.format(e))
    
    return reduced