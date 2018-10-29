import argparse
import csv
import os

# from keras import applications
# from keras.applications.resnet50 import preprocess_input
# from keras.preprocessing import image
import numpy as np
import pandas as pd
# import itertools
# import multiprocessing
# import operator
# import urllib
# import urllib.request
from concurrent.futures import ThreadPoolExecutor
from logging import DEBUG, getLogger
import os
import time

os.system('rm -f file_list.txt')
os.system('ls images/ >> file_list.txt')
pd.read_csv("file_list.txt", names=(['image'])).to_csv('data.tsv', header=True, index=True, sep='\t', index_label="id")
start_time = time.time()
os.system('python3 extract.py data.tsv')
end_time = time.time()
print("TOOK TIME : {0:d} min {1:d} sec".format(
    int(end_time - start_time) // 60, int((end_time - start_time) % 60)))
