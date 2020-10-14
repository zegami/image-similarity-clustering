# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 22:55:14 2020

@author: dougl
"""

from sklearn.manifold import TSNE
import pandas as pd


# There are many default parameters, but the ones exposed here should be
# tailored to the project specifically. These can be entered in the umap
# function below as keyword arguments. Query TSNE docstring for more options
DEFAULT_TSNE_KWARGS = {
    
    # The number of dimensions to reduce into. Typically 2, to create
    # coordinates appropriate for an X/Y graph, but can reasonably be anything
    # from 2 -> 100. For 3D plottable coordinates, use 3.
    'n_components' : 2,
    
    
    # Having a learning rate too high will cause the results to form a 'ball',
    # too low causes compressed dense clouds with few outliers.
    'learning_rate' : 200.0 # Typically 10.0 -> 1000.0
}
        

def tsne(features, write_to=None, **tsne_kwargs):
    ''' Reduces the features in the parsed pd.DataFrame 'features' into less
    dimensions (default 2). Writes the output to 'write_to' if provided, in
    .csv format. Returns the feature DataFrame.
    
    Provide any extra tsne keyword arguments as needed (query TSNE docstring
    to find these). These will override DEFAULT_TSNE_KWARGS.
    '''
    
    # Use the provided tsne arguments on top of the defaults above
    kwargs = DEFAULT_TSNE_KWARGS.copy()
    for key, val in tsne_kwargs.items():
        kwargs[key] = val
        print('[Custom TSNE argument - {}: {}]'.format(key, val))
    
    id_col_name = features.columns[0]
        
    print(f't-SNE: Reducing features to {kwargs["n_components"]} dimensions')
    
    # Don't consider the first unique ID column
    features_salient = features.copy().drop(columns=[id_col_name], axis=1)
    
    reduced = pd.DataFrame(TSNE(**kwargs).fit_transform(features_salient), dtype=object)
    reduced.insert(0, id_col_name, features[[id_col_name]])
    
    print('Success')
    
    if write_to is not None:
        try:
            reduced.to_csv(write_to, index=False)
            print('Wrote reduced features to "{}"'.format(write_to))
        except Exception as e:
            print('\nWARNING - Could not write results to file: "{}"'.format(e))
    
    return reduced