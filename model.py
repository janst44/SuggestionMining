import sys
import os
#import pandas as pd
import numpy as np
import csv
import string
import random

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # block tf from outputting anything but errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'# block os logging, it defaults to 0 (all logs shown), but can be set to 1 to filter out INFO logs, 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs.

from termcolor import colored
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from slangreplacer import translator

###################################### FUNCTIONS #######################################################
def getWordVector(X):
    global num_words_kept
    global word2vec
    global word_vec_dim

    input_vector = []
    for row in X:
        words = row.split()
        if len(words) > num_words_kept:
            words = words[:num_words_kept]
        elif len(words) < num_words_kept:
            for i in range(num_words_kept - len(words)):
                words.append("")
        input_to_vector = []
        for word in words:
            if word in word2vec:
                input_to_vector.append(np.array(word2vec[word]).astype(np.float).tolist())#multidimensional wordvecor
            else:
                input_to_vector.append([5.0] * word_vec_dim)#place a number that is far different than the rest so as not to be to similar
        input_vector.append(np.array(input_to_vector).tolist())
    input_vector = np.array(input_vector)
    return input_vector
############################################## Set up ###############################################################
shuffled = False #change to false if you want the printing of False-neg False-Pos numbers to remain with file rows
import_csv = False
play = False
test = False
default = False
load = False
cross_validate = False
debug = True
############################################# Parse Arguments #######################################################

if len(sys.argv) > 1:
    exists = os.path.isfile('./Trained.h5')
    if exists:
        user_input = input("Would you like to load the saved model? (y/n)")
        if user_input == 'y':
            load = True
    if ".csv" in str(sys.argv[1]):
        import_csv = True
        shuffled = True
        print(colored("Using", "green"), colored(str(sys.argv[1]), "green"), colored("as input", "green"))
        print(colored("Using 'Output.csv' as output", "green"))
    elif "play" == str(sys.argv[1]):
        play = True
        print(colored("Play mode selected", "green"))
    elif "test" == str(sys.argv[1]):
        test = True
        print(colored("Test mode selected", "green"))
    elif "cv" == str(sys.argv[1]):
        print(colored("Cross Validation mode selected", "green"))
        cross_validate = True
    else:
        print(colored("Error invalid argument:", "yellow"), colored(str("'" + sys.argv[1] + "'"), "yellow"))
        sys.exit()
else:
    default = True
    exists = os.path.isfile('./best_model.h5')
    if exists:
        user_input = input("Would you like to load the saved model? (y/n)")
        if user_input == 'y':
                load = True
    print(colored("Running Default Train/Test split with Statistics", "green"))
############################################# Load Data #############################################################
with open("./glove/glove.6B/glove.6B.50d.txt", "r", errors='ignore') as lines:
    word2vec = {line.split()[0]: line.split()[1:] for line in lines}
    
csv_path = "./datasets/majorityClassifier/majorityShuffled.csv"
#df = pd.read_csv(csv_path)
features = []#df.iloc[:,0]#pd.concat([df.iloc[:,0]], ignore_index=True)
labels = []#df.iloc[:,1]

if cross_validate:# Balance the Dataset
    print(colored("Resampling the Dataset to Rebalance for running Cross Validation", "green"))
    last_item = 0
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if row[1] == '0' and last_item <= 50:
                features.append(translator(row[0].translate(str.maketrans('','','!"#%\'()*+,-./:;<=>?@[\]^_`{|}~\n')).lower()))
                labels.append(row[1])
                last_item+=1
            elif row[1] == '1':
                features.append(translator(row[0].translate(str.maketrans('','','!"#%\'()*+,-./:;<=>?@[\]^_`{|}~\n')).lower()))
                labels.append(row[1])
                last_item-=1
    if last_item != 0:
        i = 0
        while last_item != 0:
            if labels[i] == '0':#adjust above value if error occurs (50)
                #print(features[i])
                del features[i]
                del labels[i]
                last_item-=1
            i+=1
else:# Dont balance the dataset
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            features.append(translator(row[0].translate(str.maketrans('','','!"#%\'()*+,-./:;<=>?@[\]^_`{|}~\n')).lower()))
            labels.append(row[1])

#print("features: ", features)
features = np.array(features)#tilt upright
labels = np.array(labels)

# Assert each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]
############################################# Shuffle ##############################################################
# Shuffle data
# Generate the permutation index array.
if shuffled:
    print(colored("Shuffling...", "green"))
    permutation = np.random.permutation(features.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    features = features[permutation]
    labels = labels[permutation]

# Create word vectors

num_words_kept = 50 #ideally use at least 500 because currently the phrase with the longest word count is just below that
word_vec_dim = 50
##################Small Test################
#labels = [1,0]
#features = ["i love cheese", "hi"]
#print(labels)
#print(features)
############################################

feature_vectors = getWordVector(features)
#print("features shape: ", np.shape(feature_vectors))

# Classifier
n_units=128 # hidden LSTM units or 9 after testing
num_classes = 2 # binary clasification
batches=16 # Size of each batch TODO: make sure each batch gets a good representation from both classes? FIXED: for the CV because cross validation does this in TF
n_epochs=100 # num times to go through training which would overfit if not for dropout 
dropout_rate = .5
########################################### Create Model ###########################################################
model = Sequential()
model.add(LSTM(n_units, return_sequences=True, input_shape=(num_words_kept, word_vec_dim)))
#model.add(Dense(4, input_dim=50, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(num_classes, activation='sigmoid'))#sigmoid because its binary classification, if multiclass output then use softmax
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

cb_list = [EarlyStopping(monitor='loss', min_delta=.1, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)]

cb_validate = [
    EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    ,ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    ]
########################################## Create Train and Test Split Depending on Args Supplied ##################
X = feature_vectors
y = labels
#print("after to array: ", np.shape(X))
#print("labels after to array: ", labels)

#X = X.reshape(1,-1,num_words_kept)
#y = y.reshape(-1,1)

if not default and not load and not cross_validate:
    print("Training...\n")
    y = to_categorical(y)#for categorical_crossentropy library
    model.fit(X, y, batch_size=batches, epochs=n_epochs, shuffle=False, callbacks=cb_list, verbose=1)
if test:
    user_input = input("Please give me a phrase to evaluate whether it is or isn't a suggestion, or enter 'quit': ")
    while user_input != 'quit':
        user_input = transator(user_input.translate(str.maketrans('','','!"#%\'()*+,-./:;<=>?@[\]^_`{|}~\n')).lower())
        #print(user_input)
        X = np.array([user_input])
        wordVector = getWordVector(X)
        pred = model.predict(wordVector)
        
        print(colored("Prediction: ", "yellow"), colored(pred, "yellow"))
        if pred[0][0] > pred[0][1]:
            print(colored("Not a suggestion!", "green"))
        else:
            print(colored("You entered a suggestion!", "green"))
        user_input = input("Give me an english sentence and i'll tell you if it is a suggestion or not, or enter 'quit': ")
elif play:
    num_games = 0
    num_games_won = 0
    if load:
        #Reload model
        model = load_model('Trained.h5')
        print(colored("MUAHAHAHA I'M BACK!!!", "yellow"))
        exists = os.path.isfile('./AIInfo.txt')
        if exists:
            f=open("./AIInfo.txt", "r")
            if f.mode == 'r':
                contents = f.readlines()#returns a list
                f.close()
                num_games = int(contents[0][0])
                num_games_won = int(contents[1][0])
        else:
            print("***Could not load meta data from previous games, resetting w/l ratio***")
    AI = 0
    human = 0
    turn = 0
    play_again = True
    while play_again != False:
        print("Number of Humans crushed so far: ", colored(num_games_won, "yellow"),"/", colored(num_games, "cyan"))
        print(colored("Mere Human challenger has fallen into my trap! You must defeat me in a game of 'Is it a Suggestion?' or get crushed! You must win 3 points before I do by correctly telling me if my sentence is a suggestion. We'll take turns. You first, give me any sentence.", "yellow"))
        while AI != 3 and human != 3:
            X
            y = None
            retrain = True
            print("\n")
            print("AI score: ", colored(AI, "yellow"), "               ", "Mere Human score: ", colored(human, "cyan"))
            if turn == 0:
                #print(colored("Your turn!", "cyan"))
                user_input = input("Enter a phrase for me to evaluate whether it is or isn't a suggestion:")
                user_input = translator(user_input.translate(str.maketrans('','','!"#%\'()*+,-./:;<=>?@[\]^_`{|}~\n')).lower())
                X = np.array([user_input])
                wordVector = getWordVector(X)
                pred = model.predict(wordVector)
                print(colored("Prediction: ", "yellow"), colored(pred, "yellow"))
                if pred[0][0] > pred[0][1]:
                    print(colored("Not a suggestion!", "green"))
                    y = '0'
                else:
                    print(colored("You entered a suggestion!", "green"))
                    y = '1'
                user_input = input("Was I correct? (y/n)")
                if user_input == 'y':
                    AI += 1
                    retrain = False
                    # Write test to file for examination
                    f = csv.writer(open('correct_new_classifications.csv','a', newline='',encoding='utf-8-sig'))
                    f.writerow([X[0],y])
                else:
                    human += 1
                    if y == '0':
                        y = '1'
                    else:
                        y = '0'
                    f = csv.writer(open('incorrect_new_classifications.csv','a', newline='',encoding='utf-8-sig'))
                    f.writerow([X[0],y])#write the correct value for using later
                turn = 1
            else:
                #print(colored("My turn!", "yellow"))
                rand = random.randint(0, np.shape(features)[0]-1)
                X = np.array([features[rand]])
                print(colored("Phrase: ", "yellow"), colored(X[0], "yellow"))
                user_input = input("Is it a suggestion? (y/n)")
                if user_input == 'y' and labels[rand] == '1':
                    print(colored("You are correct.", "cyan"))
                    human += 1
                    retrain = False
                elif user_input == 'n' and labels[rand] == '0':
                    print(colored("You are correct.", "cyan"))
                    human += 1
                    retrain = False
                else:
                    print(colored("Wrong!", "yellow"))
                    AI += 1
                    if labels[rand] == '0':
                        y = '1'
                    else:
                        y = '0'
                    f = csv.writer(open('human_incorrect_classifications.csv','a', newline='',encoding='utf-8-sig'))
                    f.writerow([X[0],y])
                turn = 0
            if retrain:
                X = getWordVector(X)
                y = np.array([y])
                y = y.reshape(-1,1)
                y = to_categorical(y, num_classes=2, dtype='float32')
                model.fit(X, y, epochs=3, batch_size=1, shuffle=False, verbose=0)

        print(colored("\n\n#####     AI score: ", "yellow"), colored(AI, "yellow"), "               ", colored("#####     Mere Human score: ", "cyan"), colored(human, "cyan"))
        if(AI == 3):
            print(colored("Hahaha, I AM THE WINNER!", "yellow"))
            num_games_won += 1
        else:
            print(colored("HAHAHA, YOUR EFFORTS ARE FUTILE! ENJOY THIS VICTORY WHILE YOU CAN HUMAN!", "yellow"), "\n")
        num_games += 1
        
        user_input = input("Would you like to play again? (y/n)")
        if user_input == 'n':
            user_input = input("Would you like to save the model? (y/n)")
            if user_input == 'y':
                #Save trained model
                model.save('Trained.h5')
                model.save('best_model.h5')
                #Save w/l ratio
                f= open("AIInfo.txt","w")
                f.write(str(num_games)+"\n")
                f.write(str(num_games_won)+"\n")
                f.close()
                print("Model Saved")
            else:
                print("Exiting without Saving")
            play_again = False
            
            del model
        else:
            AI = 0
            human = 0
            turn = 0
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
elif import_csv:
    csv_path = "./" + str(sys.argv[1])
    features = []
    with open(csv_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            features.append(translator(row[0].translate(str.maketrans('','','!"#%\'()*+,-./:;<=>?@[\]^_`{|}~\n')).lower()))
    X = np.array(features)
    wordVector = getWordVector(X)
    pred = model.predict(wordVector)
    pred = np.argmax(pred, axis=1)
    # Write test to file for classification results
    c = csv.writer(open("Output.csv", "w"))
    i=0
    label = '0'
    for p in pred:
        if p==0:
            label = '0'
        else:
            label = '1'
        c.writerow([X[i],label])
        i = i+1
elif cross_validate:
        print(colored("Running K-folds Cross-Validation", "green"))
        # fix random seed for reproducibility
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        cvscores = []
        y_temp = to_categorical(y)
        i = 1
        for train, test in kfold.split(X, y):
            #print(np.shape(X[train]), np.shape(y[test]))
            model = Sequential()
            model.add(LSTM(n_units, return_sequences=True, input_shape=(num_words_kept, word_vec_dim)))
            model.add(Dropout(dropout_rate))
            model.add(Flatten())
            model.add(Dense(num_classes, activation='sigmoid'))#sigmiod?
            model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            print(colored("Validation set", "green"), colored(i, "green"))
            model.fit(X[train], y_temp[train], validation_data=(X[test], y_temp[test]), epochs=100, batch_size=batches, verbose = 1, callbacks=cb_validate)#alternatively you can pass in the validation set with the validation_data=(x_test, y_test) and use a callbacks= "val_loss"
            scores = model.evaluate(X[test], y_temp[test], verbose=0)
            #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            i+=1
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
else:
    split = int(np.shape(feature_vectors)[0]*.1)
    print("Dataset size: ", int(np.shape(feature_vectors)[0]))
    print("Split: ", split)
    y = to_categorical(y)#for categorical_crossentropy library
    X_train, X_test = X[split:], X[:split]#train takes the last larger% of the data, test takes first smaller % of the data so i can better see the corresponding sentences in the csv and not have to use offset.
    y_train, y_test = y[split:], y[:split]
    #print("X_train: ", np.shape(X_train))
    #print(X_train)
    #print("X_test: ", np.shape(X_test))
    #print(X_test)
    #print("y_train: ", np.shape(y_train))
    #print("y_test: ", np.shape(y_test))
    #print(y_train)
    #print(y_test)

    ############################################ Train model ###########################################################
    model.summary()
    
    if not load:
        model.fit(X_train, y_train, validation_split=0.3, batch_size=batches, epochs=n_epochs, shuffle=False, callbacks = cb_validate)
        model = load_model('best_model.h5')
    else:
        model = load_model('best_model.h5')
    # Predict
    test_loss = model.evaluate(X_test, y_test)
    print("Test accuracy and loss:      loss: ", test_loss[0], "     accuracy: ", test_loss[1], "\n")


    # Statistics to evaluate model
    y_true = np.argmax(y_test,axis=1)
    pred = np.argmax(model.predict(X_test), axis=1)
    print("Confusion Matrix:\n[[TN, FP]\n[FN, TP]]")
    print(confusion_matrix(y_true, pred))
    print(classification_report(y_true, pred))

    # Show misclassifed suggestions True negative and False positive
    if debug:
        print("Mis-Classified data and their index:")
        suggestion_index = 0 # this is the index that will correspond to the line in the SuggestionTest.csv created below, not the original data file because of the shuffle if true
        for p in pred:
            if p == 1 and labels[suggestion_index] == '0':#False positive (Model thought it was true, but it was actually false)
                print("FP Suggestion:", suggestion_index+1)#index in csv file is not 0 based
                pass
            elif p == 0 and labels[suggestion_index] == '1':#False negative (Model thought it was false, but it was actually true)
                print("FN Suggestion:", suggestion_index+1)
                pass
            suggestion_index +=1
        # Write test to file for examination
        #c = csv.writer(open("SuggestionTest.csv", "w"))
        #i=0
        #for x in features[:split]:
            #c.writerow([features[i],labels[i]])
            #i = i+1

print("Done!")


#precision - If you consistently measure your height as 5’0″ with a yardstick, your measurements are precise. Even if you are in face 5'5"
#recall - A number of events you can correctly recall = True positive (they’re correct and you recall them)
#f1-score - harmonic mean of precision and recall
#support - The support is the number of occurrences of each class in y_true.
