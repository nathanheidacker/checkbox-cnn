A CNN for classifying images of virtual checkboxes

v1 BEST: 82%
v2 BEST: 83.5%

weights: https://drive.google.com/drive/folders/12WSQ4Nt_Atw3eArEQzovBqViyTrb5CDH?usp=sharing

# Checkbox CNN
This repository contains the training and inference files for a series of convolutional neural networks designed to classify images of html checkboxes.

The contents of this readme are organized as follows:
    TOC HERE

## The Assignment
Given a dataset of ~500 png image files categorized as "checked", "unchecked" or "other", train a classifier that can classify a novel image into one of these three categories. This assignment was provided as part of an interview process, and as such, I no longer have access to the dataset used in training the model. If you'd like to recreate the training portion of the model, you must procure access to the dataset or to another similar dataset on your own.

## Setup
There are three ways to get setup with this project, which will be described in order of increasing difficulty. All three methods require some initial setup depending on what the user intends to test.

### Initial Setup
To begin with the project, clone this repository and download the weights from this google drive link. Both of these binary files should be placed inside of the 'weights' directory at the root of the repository, and will be used as the trained model weights for the respective model versions. This is the only component of the manual setup that is absolutely necessary.

If the user also wants to be able to train the model from scratch, they must download the compressed dataset file. The expanded contents of this compressed file (a directory called 'data', with subdirectories 'checked', 'unchecked' and 'other', each containing a sequence of PNG images) should be placed in the root directory of the cloned repository. These files comprise the data that will be used for training and validation of the model, on which the CheckboxData class relies. If the user does not intend to train the model, these files are technically unnecessary.

Lastly, if the user wants to 'infer', or test the model's predictive ability on a new image, these images should be copied into 'tests' directory contained within the root directory. Some examples of test cases have already been provided (2 checked, 2 unchecked).

### Option 1: Bash scripts
If bash is available in the testing environment, simply run `bash setup.sh` with the current working directory as the root directory of the cloned repository. This will automatically build a Docker image of the project and spin up a docker container of the image, as well as open an interactive terminal so that commands can be executed. If you quit the terminal, you can reopen it without recreating the Docker image by running `bash run.sh` instead.

### Option 2: Manually create docker container
If you'd like to create a Docker environment manually, you can do with the provided Docker file.

### Option 3: Create a new virtual environment
If neither of the above options are available, you can create a new python virtual environment (or do so in conda), and pip install the provided requirements.txt

## Testing the Project
If you setup the project using option 3, change your current working directory to be the root directory of the project. If you setup the project using options 1 or 2, you should be at the root directory of the project by default when spinning up the docker container. All of the commands exposed by the project should be run at this top-level directory.

Four python scripts are available to be used in the /src directory. They are:
 - data.py
 - model.py
 - train.py
 - evaluate.py

### src/data.py
This script contains a class called CheckboxData, which is a dataloader initializer that is used in the training and evaluation of the CNN models. When run directly via with an integer argument, it will print some stats about the dataset, where the argument is used to determine the number of each class that should be present.

### src/evaluate.py

### src/train.py

### src/model.py

## CheckboxCNN v1

### Approach

### Data Preparation

### Model Design

### Results

### Visualization

### Improvements

## CheckboxCNN v2

### Approach

### Data Preparation

### Model Design

### Results

### Visualization

### Improvements