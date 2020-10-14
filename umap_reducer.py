# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 00:32:06 2020

@author: dougl
"""

from umap import UMAP
import pandas as pd


# There are many default parameters, but the ones exposed here should be
# tailored to the project specifically. These can be entered in the umap
# function below as keyword arguments
DEFAULT_UMAP_KWARGS = {
    
    # Use 'categorical' for categorised data not on a continuum, like types
    # of plants, etc. 'l1' or 'l2' are more appropriate for continuous values,
    # like the feature vectors of a model
    'target_metric' : 'l1' # 'categorical' 'l1' 'l2'
}


def umap(features, dims=2, write_to=None, **umap_kwargs):
    ''' Reduces the features in the parsed pd.DataFrame 'features' into 'dims'
    dimensions (default 2). Writes the output to 'write_to' if provided, in
    .csv format. Returns the feature DataFrame.
    '''
    
    # Use the provided umap arguments on top of the defaults above
    kwargs = DEFAULT_UMAP_KWARGS.copy()
    for key, val in umap_kwargs.items():
        kwargs[key] = val
        print('[Custom UMAP argument - {}: {}]'.format(key, val))
    
    if dims != 2:
        print('UMAP: Not currently supporting anything but 2-dim reduction')
    
    id_col_name = features.columns[0]
        
    print('UMAP: Reducing features to {} dimensions'.format(dims))
    
    # Don't consider the first unique ID column
    features_salient = features.copy().drop(columns=[id_col_name], axis=1)
    
    reduced = pd.DataFrame(UMAP(**kwargs).fit_transform(features_salient), dtype=object)
    reduced.insert(0, id_col_name, features[[id_col_name]])
    
    print('Success')
    
    if write_to is not None:
        try:
            reduced.to_csv(write_to, index=False)
            print('Wrote reduced features to "{}"'.format(write_to))
        except Exception as e:
            print('\nWARNING - Could not write results to file: "{}"'.format(e))
    
    return reduced
