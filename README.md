# Unsupervised ML with Keras pre-trained models and t-SNE
This project allows images to be automatically grouped into like clusters using a combination of machine learning techniques.

Unsupervised machine learning is a technique that can used to automatically classify or group data together that has no preidentified structure (as opposed to supervised learning where an "expert" has labeled a training data set).

There are two steps involved in the process. The first is to use a pre-trained deep learning model to extract a **feature vector** of each image in the collection. Once we have the vector (which is an array of floating point values) it is then be passed into a t-SNE function, which takes all of the arrays and reduces them down to two values: **X** and **Y**. These two values can then be plotted against each other to produce a graph which Zegami can use as a filter.

## extract.py
Uses one of the [pre-trained deep learning models avaliable in Keras](https://keras.io/applications) to extract a feature vector for all images in a source directory.

I used the following guide to [install Keras with TensorFlow](https://keras.io/#installation) using conda.

The script expects as an argument the path to a tab separated file that has at a minimum a column called 'id' and another called 'image' which contains the file name. The images **need** to be located in a directory called **images** which is located in the same directory as the source file.

For example if a file called example.tsv contains a single record:

| id | image |
| -- |:-----:|
| 1  | 1.jpg |

Then it would have the following directory structure:
```
.
+-- example.tsv
+-- images
|   +-- 1.jpg
```

The results are saved to a tab separated file postfixed with '_features.

## tsne.py
Takes a comma separated list of values and runs them through a t-SNE function. The result is then saved back to a tab separated file postfixed with '_tsne'.
