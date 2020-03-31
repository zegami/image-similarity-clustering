# Unsupervised feature extraction and reduction
This project allows numerical features to be reduced down to fewer dimensions for plotting using unsupervised machine learning.
Features can be taken simply as face value numbers from a spreadsheet (csv) file, or they can be extracted from images using a pre-trained model.

## CLI
All functions in this package can be imported for use in your own python scripts, or run as stand-alone commands in a CLI.

In order to deal with all inputs in a standardardised fashion, csv files are parsed using `parse_data` in `parse_data.py`.
While this is done automatically for CLI commands, if you're writing your own scripts you should parse your csv data in
through this first. It essentially puts your data in a pd.DataFrame, where the first column is always a unique ID key column.

Current functionality:
- `python cli.py features <image-or-directory-of-images> <output-csv-path>`
- `python cli.py tsne <image-or-directory-of-images> <output-csv-path> <feature-cols> <unique-col>`
- `python cli.py umap <image-or-directory-of-images> <output-csv-path> <feature-cols> <unique-col>`

Running 'features' will extract the numerical features of a directory of images, and save them (with the unique IDs) to the output path.

Running 'tsne' or 'umap' will reduce such features (or features from a regular csv) into fewer dimensions, and save these (with the unique IDs) to the output path.
These reduction functions will accept a `--model` argument, allowing you to specify one of several common pre-trained models to be used. I will probably add a command
to specify your own custom model soon.

*It is worth reiterating:*

As stated in the above, it is imperetive you parse any data you want to reduce using `parse_data` first if you're accessing these functions in your own scripts, and not using the CLI.

## Additional
This project uses keras and tensorflow 2. I run these commands using tensorflow-gpu, so for the smoothest experience I recommend setting CUDA up from NVIDIA's website, I have not tested regular tensorflow (cpu).

# Example usage

## Reducing a folder of images using t-SNE
I have some folder full of images at path `./images` that I want extract features from, and reduce into 2 dimensions using t-SNE.

### CLI
```
python cli.py features "./images" "features.csv"
python cli.py tsne "features.csv" "tsne-results.csv" --feature-cols all --unique-col A
```

### Scripting
```
from features import extract_features
from tsne_reducer import tsne

features = extract_features('./images')
reduced = tsne(features, write_to='./tsne_features.csv')
```

## Reducing some numerical data columns using UMAP
I have a csv file called `data.csv` containing a unique ID column (called "name") at column D, and the important columns containing numbers are at A, C, H, and AB.

### CLI
```
python cli.py umap "./data.csv" "./tsne.csv" --feature-cols A,C,H,AB --unique-col D
```

### Scripting
```
from parse_data import parse_data
from umap_reducer import umap

data = parse_data('./data.csv', feature_cols=['A', 'C', 'H', 'AB'], unique_col='D')
reduced = umap(data, write_to='./umap_features.csv')
```


