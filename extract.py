#!/usr/bin/env python3
#
# Copyright 2017 Zegami Ltd

"""Preprocess images using Keras pre-trained models."""

import argparse
import csv
import os

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')


parser = argparse.ArgumentParser(prog='Feature extractor')
parser.add_argument('source', default=None, help='Path to the source metadata file')
parser.add_argument('model', default='ResNet50', type=named_model, help='Name of the pre-trained model to use')
parser.add_argument('--delimiter', default='\t', help='File delimiter to use, defult tab')

pargs = parser.parse_args()

source_dir = os.path.dirname(pargs.source)


def get_feature(metadata):
    print('{}'.format(metadata['id']))
    try:
        img_path = os.path.join(source_dir, 'images', metadata['image'])
        if os.path.isfile(img_path):
            print('is file: {}'.format(img_path))
            try:
                # load image setting the image size to 224 x 224
                img = image.load_img(img_path, target_size=(224, 224))
                # convert image to numpy array
                x = image.img_to_array(img)
                # the image is now in an array of shape (3, 224, 224)
                # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # extract the features
                features = pargs.model.predict(x)[0]
                # convert from Numpy to a list of values
                features_arr = np.char.mod('%f', features)

                return {"id": metadata['id'], "features": ','.join(features_arr)}
            except Exception as ex:
                # skip all exceptions for now
                print(ex)
                pass
    except Exception as ex:
        # skip all exceptions for now
        print(ex)
        pass
    return None


def start():
    try:
        # read the source file
        data = pandas.read_csv(pargs.source, sep=pargs.delimiter)

        # extract features
        features = map(get_feature, data.T.to_dict().values())

        # remove empty entries
        features = filter(None, features)

        # write to a tab delimited file
        source_filename = os.path.splitext(pargs.source)[0].split(os.sep)[-1]
        ending = "tsv"
        if pargs.delimiter == ",":
            ending = "csv"
        with open(os.path.join(source_dir, '{}_features.{}'.format(source_filename, ending)), 'w') as output:
            w = csv.DictWriter(output, fieldnames=['id', 'features'], delimiter=pargs.delimiter, lineterminator='\n')
            w.writeheader()
            w.writerows(features)

    except EnvironmentError as e:
        print(e)


if __name__ == '__main__':
    start()
