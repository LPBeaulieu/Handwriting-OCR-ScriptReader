#NOTE: This code is by no means optimized in terms of operating speed, contains
#many nested foor loops that could in principle be handled with Numpy, but since
#you will likely use this with short word lists of around 20,000 words or less to
#minimize the chance of random words being chosen instead of your desired corrections,
#it should be done running within 30 minutes. The abridged dictionary runs very
#quickly with "get_predictions.py", so the slow running time of "dictionary.py" isn't
#an issue, all things considered.

import csv
import os
import random
import glob
from alive_progress import alive_bar


problem = False
cwd = os.getcwd()

#The 27K word list ("wlist_match7.txt") was found at the following link
#(https://www.keithv.com/software/wlist/) and it was assembled by selecting
#words at the intersection of 12 different word lists, such as the
#British national corpus.

#The dictionary text file is retrieved using the "glob" module.
#If there is more than one text file within the "Dictionary" subfolder
#of the working folder, an error message is displayed in the Powershell.
#The same is true if no TXT file is found within the folder.
path_txt = os.path.join(cwd, "*.txt")
txt_files = glob.glob(path_txt)
if txt_files == []:
    print("\nPlease include a text file (.txt) containing " +
    'the list of words that you wish to use in the "Dictionary" subfolder ' +
    'of the working folder. Its name must not start with "Abridged", as ' +
    'this prefix will be added to the abridged version of the dictionary, ' +
    'without duplicates.')
    problem = True
elif len(txt_files) > 1:
    print("\nPlease include a text file (.txt) containing " +
    'the list of words that you wish to use in the "Dictionary" subfolder ' +
    'of the working folder. Its name must not start with "Abridged", as ' +
    'this prefix will be added to the abridged version of the dictionary, ' +
    'without duplicates.')
    problem = True
else:
    txt_file_path = txt_files[0]
    txt_file_name = os.path.basename(txt_file_path)

if problem == False:
    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.shuffle(lines)

    english_dict = []
    duplicate_words = []
    with alive_bar(len(lines)) as bar:
        #For every word in the starting dictionary (at index "i"), the variable "append_ok" starts
        #off being "True", meaning that the word could be appended to the list "english_dict" if
        #no word bearing the same sequence of letters save one (henceforth termed duplicates)
        #is found within the unabridged dictionary.
        for i in range(len(lines)):
            append_ok = True
            #Only the words having four or more letters and not comprising an apostrophe are selected
            #from the unabridged word list for screening for duplicates.
            if len(lines[i].strip()) > 3 and lines[i].strip() not in english_dict and "'" not in lines[i]:
                #Every letter index within the word is substituted with another letter than the letter found
                #at that index within the original word. If the modified word is found within "english_dict",
                #it means that the substitution of one letter within the original word leads to the formation
                #of another word within the unabridged dictionary (a duplicate), and therefore both words
                #must be removed at some point to avoid correcting words that have more than one possible
                #correct spelling at a given index. Both words will then be added to the list "duplicate words".
                #The variable "append_ok" is then set to "False", to avoid the original word under investigation
                #from being included in the list "english_dict".
                for j in range(len(lines[i].strip())):
                    for letter in ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
                    "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]:
                        if j == 0 and letter != lines[i][j] and letter + lines[i][1:].strip() in english_dict:
                            word_under_investigation = lines[i].strip()
                            duplicate_word = letter + lines[i][1:].strip()
                            if duplicate_word not in duplicate_words:
                                duplicate_words.append(duplicate_word)
                            if word_under_investigation not in duplicate_words:
                                duplicate_words.append(word_under_investigation)
                            append_ok = False
                        elif (j < len(lines[i].strip())-2 and j > 0 and letter != lines[i][j] and
                        lines[i][:j].strip() + letter + lines[i][j+1:].strip() in english_dict):
                            word_under_investigation = lines[i].strip()
                            duplicate_word = lines[i][:j].strip() + letter + lines[i][j+1:].strip()
                            if duplicate_word not in duplicate_words:
                                duplicate_words.append(duplicate_word)
                            if word_under_investigation not in duplicate_words:
                                duplicate_words.append(word_under_investigation)
                            append_ok = False
                        elif (j == len(lines[i].strip())-2 and letter != lines[i][j] and
                        lines[i][:j].strip() + letter + lines[i][j+1].strip() in english_dict):
                            word_under_investigation = lines[i].strip()
                            duplicate_word =  lines[i][:j].strip() + letter + lines[i][j+1].strip()
                            if duplicate_word not in duplicate_words:
                                duplicate_words.append(duplicate_word)
                            if word_under_investigation not in duplicate_words:
                                duplicate_words.append(word_under_investigation)
                            append_ok = False
                        elif (j == len(lines[i].strip())-1 and letter != lines[i][j] and
                        lines[i][:j].strip() + letter in english_dict):
                            word_under_investigation = lines[i].strip()
                            duplicate_word =  lines[i][:j].strip() + letter
                            if duplicate_word not in duplicate_words:
                                duplicate_words.append(duplicate_word)
                            if word_under_investigation not in duplicate_words:
                                duplicate_words.append(word_under_investigation)
                            append_ok = False
                        elif append_ok == True:
                            if lines[i].strip() not in english_dict:
                                english_dict.append(lines[i].strip())
            bar()

    #The abriged list of words is assembled by list comprehension, by cycling through
    #the words in "english_dict" and removing those that are also found in the
    #"duplicate_words" list. The list is sorted so that the final abridged word list
    #is in alphabetical order.
    english_dict = [w.strip() for w in english_dict if w not in duplicate_words]
    english_dict.sort()

    #The abridged word list (without duplicates) is saved as a text file,
    #with each word occupying one line, and with the file name starting with
    #Abridged. #The code "get_predictions.py" will automatically retrieve the
    #original word list and its abridged form from the "Dictionary" subfolder
    #within the working foler.
    with open(os.path.join("Abridged " + txt_file_name), "w", encoding="utf-8") as g:
        g.writelines(f'{word}\n' for word in english_dict)
