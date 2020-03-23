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
