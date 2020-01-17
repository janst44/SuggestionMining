# SuggestionMining

Tells you whether or not a sentence/phrase/clause is a suggestion. Doesn't heed punctuation.

## Before you run the model

This repo requires the Glove dataset. You can download on from: https://nlp.stanford.edu/projects/glove/. I used the glove.6B.zip. Extract it into the root directory and it should be in /glove folder for script to find it.

### Installation

If using python3 you can use pip3 instead of pip below as needed:
```
pip install numpy
pip install keras
pip install scikitlearn
pip install tensorflow
```

## How to run the model

    python3 model.py
This will run the model in default train/test split mode

Output: Statistics on how well it did to the command line. If debug in the program is manually set to 'True' then this mode also outputs all FN and FP classified instances and their line in the input file which is found in the datasets folder called "majorityShuffled.csv".


## Optional Arguments:

    python model.py play
An interactive play mode that continuously trains the model, the model is saved upon quitting (It's fun and a great way to train, its actually most important mode of all)

Output: A saved model that can be imported on future runs and improved with each run it is included. "Trained.h5". Also outputs all misclassified and correctly classified instances in their respective files.




    python model.py test
An interactive testing ground to manually feed the model single sentences (must be shorter than 100 words)

Output: A single classified instance and its belief probability




    python model.py *.csv
Takes any .csv file as input (one sentence per row in the first column of the file)

Output: A csv file with the second column classified based off the model that was trained "Output.csv"




    python model.py cv
Runs Cross-Validation and reports average accuracy on the training set for 5 folds

Output: average accuracy and error on the validation sets as well and the standard deviation +-(best and worse cases)
    

## Authors

* **Joshua Campbell**
* **Alexander Hamilton**
* **Cameron Timpson**
* **Areuni Anderson**




