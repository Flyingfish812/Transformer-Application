# Transformer-Application

`my_utils.py` is the file that stores the tools (data generating, training, testing)

`presets.py` is the file that builds the model and keep the results.

`main.py` is the training codes.

`test.py` is the testing codes.

`test.ipynb` serves as a draft, and an easy approach of visualization.

`config` folder saves the config files (in yaml) that can be used as training / testing configuration.

`result` folder saves the result for each test.

The models' parameters will be saved in a new folder named `model`.

`run.bat` is a batch file that can automatically run the training and testing, you can change the commands in this file to run the code on different configs.

## PFRTool Library

This library contains all the useful functions used in this project

Usage:

1. Define a config file (example format in: `config/config_pack_vit_1.yaml`)

2. Define a main file (example format in: `main_pack.py`)

3. Run the file in command line (example: `python main_pack.py config/config_pack_vit_1.yaml`), or use the batch/bash file (example format in: `run.bat`)

## Data Availability

[NOAA sea surface temperature set](https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu)