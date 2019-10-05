#!/usr/bin/env python3
#
# Copyright 2017 Zegami Ltd

"""Perform t-SNE on a preprocessed dataset."""

import argparse
import csv
import os
import sys

import numpy
import pandas
import umap


def process(data, reducer):
    # split the comma delimited string back into a list of values
    transformed = [d.split(',') for d in data['features']]

    # convert image data to float64 matrix. float64 is need for bh_sne
    x_data = numpy.asarray(transformed).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform umap
    vis_data = reducer.fit_transform(x_data)

    # convert the results into a list of dict
    results = []
    for i in range(0, len(data)):
        results.append({
            'id': data['id'][i],
            'x': vis_data[i][0],
            'y': vis_data[i][1]
        })
    return results


def write_tsv(results, output_tsv):
    # write to a tab delimited file
    with open(output_tsv, 'w') as output:
        w = csv.DictWriter(
            output, fieldnames=['id', 'x', 'y'], delimiter='\t',
            lineterminator='\n')
        w.writeheader()
        w.writerows(results)


def main(argv):
    parser = argparse.ArgumentParser(prog='UMAP')
    parser.add_argument('source', help='path to the source metadata file')
    parser.add_argument(
        '-l', '--limit', type=int, help='use subset of first N items')
    args = parser.parse_args(argv[1:])

    # read in the data file
    data = pandas.read_csv(args.source, sep='\t')
    if args.limit:
        data = data.iloc[:args.limit]

    reducer = umap.UMAP()
    results = process(data, reducer)

    destination_dir = os.path.dirname(args.source)
    source_filename = os.path.splitext(args.source)[0].split(os.sep)[-1]
    tsv_name = os.path.join(destination_dir, '{}_umap.tsv'.format(
        source_filename))

    write_tsv(results, tsv_name)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
