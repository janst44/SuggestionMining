import csv
import re

def translator(user_string):
    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "slang.csv"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in column[0] and phrases in column[1]
            dataFromFile = csv.reader(myCSVfile, delimiter=",")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for column in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.lower() == column[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = column[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    return ' '.join(user_string)

