#!/bin/bash

# Set working directory to root
cd "$(dirname "$0")"

# Crate data directory if it does not already exist
mkdir -p data

# TODO: These need updating to follow the new structure of: data/dataset_name/images/*.png
kaggle datasets download jessicali9530/celeba-dataset -p data/
unzip data/celeba-dataset.zip -d data/celeba
rm data/celeba-dataset.zip

kaggle competitions download -c gan-getting-started -p data/
unzip data/gan-getting-started.zip -d data/art_dataset
rm data/gan-getting-started.zip
