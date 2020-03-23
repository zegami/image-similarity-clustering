# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:17:54 2020

@author: Doug Lawrence
"""

import argparse
import sys
import os
from parse_data import parse_data
from features import extract_features
from tsne_reducer import tsne
from umap_reducer import umap


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', help='extract | tsne | umap')
    parser.add_argument('data', help='[features]: Filepath to an image or folder containing images to extract features from. [tsne/umap]: Filepath to a .csv file to read into a DataFrame. ')
    parser.add_argument('out', help='Output filepath of operation')
    parser.add_argument('--feature-cols', '-f', help='[tsne/umap]: Numerical data column indices to treat as features. Ex: "B,C,F", use "all" to consider all columns (excluding optional unique-col).')
    parser.add_argument('--unique-col', '-u', help='[tsne/umap]: The column index containing unique IDs for each row (typically "ID" or "Name" column). Not required. Omitted from "all" feature_cols')
    parser.add_argument('--reduce', '-r', help='[tsne/umap]: How many dimensions to reduce features to. Default is 2.', default='2')
    parser.add_argument('--model', '-m', help='[features]: Which model to use. ResNet50 | Xception | VGG16 | VGG19 | InceptionV3 | MobileNet. Default: ResNet50', default='ResNet50')
    
    args = parser.parse_args(argv[1:])
    
    
    # === FEATURE EXTRACTION ===
    # We expect an image filepath or folder of images
    if args.mode == 'features':
        assert os.path.exists(args.data),\
            'Features mode (data arg): File or directory not found: "{}"'\
            .format(args.data)
            
        # Calculate and write to args.out
        features = extract_features(args.data, model=args.model, write_to=args.out)
            
        
    # === DIMENSION REDUCTION ===
    # We expect a .csv file of features
    elif args.mode in ['tsne', 'umap']:
        
        # Make sure we know what columns are intended to be used numerically as a list of strings, or 'all'
        feature_cols = args.feature_cols
        if feature_cols is None:
            raise Exception('Feature reduction mode: No data column indices provided. Example usage: "--feature_cols B,C,F", "--feature_cols all"')
        elif feature_cols != 'all':
            feature_cols = [s.strip() for s in feature_cols.split(',') if s.strip() != '']
        
        # Parse the data into a squashed pd.DataFrame with first column being unique keys
        df = parse_data(args.data, feature_cols, args.unique_col)
        
        if args.mode == 'tsne':
            tsne(df, dims=int(args.reduce), write_to=args.out)
            
        elif args.mode == 'umap':
            umap(df, write_to=args.out)
    

if __name__ == '__main__':
    sys.exit(main(sys.argv))