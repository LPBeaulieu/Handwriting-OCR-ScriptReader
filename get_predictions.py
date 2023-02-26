#***IMPORTANT NOTE!!***
#You cannot start your actual JPEG file names with "back",
#as the presence of the "back" prefix in the file name
#designetes whether the page is even or odd numbered, with the
#file names for even-numbered pages starting with "back".

from fastai.vision.all import *
import cv2
import os
import shutil
import re
import numpy as np
import sys
import math
from textblob import Word
from contextlib import contextmanager
import pathlib
import platform
import zipfile


if __name__ == '__main__':
    cwd = os.getcwd()

    #Import the convoluted neural network (cnn) deep learning model for OCR prediction.
    #My optimal model trained on 26 pages of typewritten text using a 1968 Olivetti Underwood Lettra 33 typewriter,
    #with a batch size of 64, a learning rate of 0.005 and 3 epochs of training yielded a validation accuracy
    #of 99.96%. The average validation accuracy for 4 training runs with the same hyperparameters mentioned above
    #is 99.90% and the optimal model (99.96%) was selected and named Model_Olivetti_1968_Underwood_Lettra_33_acc9996
    def find_model_name(model_names):
        if model_names == []:
            print("\nPlease include a CNN model zipped folder (.zip) in the working folder.")
        elif len(model_names) > 1:
            print("\nPlease include a single CNN model zipped folder (.zip) in the working folder. " +
            "Also remember to delete the github zipped project folder after extracting it in your working folder.")
        elif len(model_names) == 1:
            model_name = os.path.basename(model_names[0])
            if model_name != "Handwriting-OCR-ScriptReader-main":
                return model_name
        else:
            print("\nPlease include a single CNN model zipped folder (.zip) in the working folder. " +
            "Also remember to delete the github zipped project folder after extracting it in your working folder.")

    if platform.system() == "Windows":
        try:
            posix_backup = pathlib.PosixPath
            pathlib.PosixPath = pathlib.WindowsPath
            #The list "model_names" is populated with the ".zip" file names in
            #the working folder, if the file is recognized as a zip file by the
            #zipfile module, and if the file doesn't have an extension, when
            #the "os.path.splitext()" method is applied to the file name. It would
            #then give an empty string for zip files, but the ".odt" extensions wouldn't
            #be retained in the list, although for some reason they pass the zipfile test.
            model_names = ([file_name for file_name in sorted(os.listdir(cwd)) if
            zipfile.is_zipfile(file_name) and os.path.splitext(file_name)[-1] == ""])
            model_name = find_model_name(model_names)
            if model_name != None:
                learn = load_learner(model_name)
        finally:
            pathlib.PosixPath = posix_backup
    else:
        model_names = ([file_name for file_name in sorted(os.listdir(cwd)) if
        zipfile.is_zipfile(file_name) and os.path.splitext(file_name)[-1] == ""])
        model_name = find_model_name(model_names)
        if model_name != None:
            learn = load_learner(model_name)

    if model_name != None:
        #The list "JPEG_file_names" is populated with the ".jpg" file names in
        #the "OCR Raw Data" folder.
        front_JPEG_file_names = ([file_name for file_name in sorted(os.listdir(os.path.join(cwd,
        "OCR Raw Data"))) if (file_name[:4].lower()!="back" and file_name[-4:] == ".jpg")])

        #The code is generating cropped character images from the image files listed
        #in the "JPEG_file_names" list and store them in an image folder. These cropped
        #character images will be deleted further on in the code (see comments below)
        #and the image folder name is extracted from the first image name in the
        #"JPEG_file_names" list, including all characters up to the last hyphen
        #(e.g. "Alice's Adventures in Wonderland Chapter 1-0001.jpg" would
        #give the following extracted name: "Alice's Adventures in Wonderland Chapter 1")

        if front_JPEG_file_names not in [None, []]:
            hyphen_matches = re.finditer("-", front_JPEG_file_names[0])
            hyphen_indices = []
            for match in hyphen_matches:
                hyphen_indices.append(match.start())
            OCR_text_file_name = front_JPEG_file_names[0][:hyphen_indices[-1]]


        #The list "back_JPEG_file_names" is populated with the ".jpg" file names in
        #the "OCR Raw Data" folder.
        back_JPEG_file_names = ([file_name for file_name in sorted(os.listdir(os.path.join(cwd,
        "OCR Raw Data"))) if (file_name[:4].lower()=="back" and file_name[-4:] == ".jpg")])

        if back_JPEG_file_names not in [None, []]:
            #The user has simply reversed the stack of papers into the multi-
            #page scanner, resulting in the page numbers being reversed.
            #The "sort()" method will put them in the correct sequence.
            back_JPEG_file_names.sort(reverse=True)
            hyphen_matches = re.finditer("-", back_JPEG_file_names[0])
            hyphen_indices = []
            for match in hyphen_matches:
                hyphen_indices.append(match.start())
            OCR_text_file_name = back_JPEG_file_names[0][4:hyphen_indices[-1]].strip()

        #The folder "OCR Predictions" is created in the working folder, if
        #not already present.
        path = os.path.join(cwd,  "OCR Predictions",  OCR_text_file_name)
        if not os.path.exists(path):
            os.makedirs(path)

        #The list "JPEG_file_name" is populated with the paths for the scan images,
        #intercalating the back pages after the front pages (if applicable).
        JPEG_file_names = []
        if (front_JPEG_file_names not in [None, []] and back_JPEG_file_names not in [None, []] and
        len(front_JPEG_file_names) >= len(back_JPEG_file_names)):
            for i in range(len(front_JPEG_file_names)):
                JPEG_file_names.append(front_JPEG_file_names[i])
                try:
                    JPEG_file_names.append(back_JPEG_file_names[i])
                except IndexError:
                    pass
        elif (front_JPEG_file_names not in [None, []] and back_JPEG_file_names not in [None, []] and
        len(front_JPEG_file_names) < len(back_JPEG_file_names)):
            for i in range(len(back_JPEG_file_names)):
                try:
                    JPEG_file_names.append(front_JPEG_file_names[i])
                except IndexError:
                    pass
                JPEG_file_names.append(back_JPEG_file_names[i])

        elif front_JPEG_file_names not in [None, []] and back_JPEG_file_names in [None, []]:
            for i in range(len(front_JPEG_file_names)):
                JPEG_file_names.append(front_JPEG_file_names[i])

        elif front_JPEG_file_names in [None, []] and back_JPEG_file_names not in [None, []]:
            for i in range(len(back_JPEG_file_names)):
                JPEG_file_names.append(back_JPEG_file_names[i])



        print("\nCurrently processing a total of " + str(len(JPEG_file_names)) +
        ' JPEG scanned images of handwritten text. ' +
        'For best results, these should be scanned as JPEG images on a ' +
        'multipage scanner at a resolution of 300 dpi with US Letter paper size setting.\n')

        #The number of inches (generated with PrintANotebook)
        #in-between every dot will allow the code to determine
        #the dot spacing in pixels, assuming that the pages
        #were printed with no scaling.
        inches_between_dots = None
        dot_diameter_pixels = 5
        x_overlap = None
        y_overlap = None
        #The "top_margin_y_pixel" maps to the "y" pixel
        #where the lines or dots start being drawin on
        #the pages.
        top_margin_y_pixel = 0.95*300
        #Similarly, the "bottom_margin_y_pixel" maps to
        #the "y" pixel where the lines and dots end.
        bottom_margin_y_pixel = 2550-(0.60*300)
        #The variables "left_margin_x_pixel"  and
        #"right_margin_x_pixel" map to the "x"
        #pixels where the lines and dots start and
        #stop being drawn on the pages, respectively.
        left_margin_x_pixel = 0.25*300
        #The right margin is different from PrintANotebook,
        #as only a half letter page is scanned at a time.
        right_margin_x_pixel = 3300/2-(0.25*300)
        #The "gutter_margin_width_pixels" designates the
        #width (in pixels) of the gutter margins of the
        #notebook. They are set to the pixel equivalent
        #of an eighth of an inch, so they won't be noticeable
        #when opening a bound book.
        gutter_margin_width_pixels = 0.75*300
        #The user needs to provide the "x,y" coordinates of the
        #to upper corner dots of the first page on the stack of
        #scanned pages, so that the code could know where to
        #perform character segmentation.
        top_left_dot = None
        top_right_dot = None
        #If the user has selected the "basic_autocorrect" option, then the code
        #will screen the misspelled words in the page (words that aren't found
        #within "english_dict") to see if any of them could be corrected by
        #altering one letter, and that the resulting word could be found in
        #the "short_english_dict". If so, the substitution would be made.
        #Also, the corrected word would be in uppercase if every letter
        #of the misspelled word was in uppercase as well. If only the first
        #letter of the misspelt word was in uppercase, then the corrected
        #word would be capitalized. Otherwise, the corrected word will be
        #in lowercase.
        basic_autocorrect = False
        #If the user has selected the "basic_autocorrect_lower" option, then the code
        #will screen the misspelled words in the page (words that aren't found
        #within "english_dict") to see if any of them could be corrected by
        #altering one letter, and that the resulting word could be found in
        #the "short_english_dict". If so, the substitution would be made.
        #Also, the corrected word would only be in uppercase if every letter
        #of the misspelled word was in uppercase as well. Otherwise, the
        #corrected word will be in lowercase.
        basic_autocorrect_lower = False
        #If the user has selected the "autocorrect" option, then the code
        #will screen all of the words against the TextBlob "spellcheck()" method,
        #and substitutions will only be done if the confidence is above 90% that
        #the suggestion is the correct one. Also, the corrected word would be
        #in uppercase if every letter of the misspelled word was in uppercase
        #as well. If only the first letter of the misspelt word was in uppercase,
        #then the corrected word would be capitalized. Otherwise, the corrected
        #word will be in lowercase. The user may specify a different autocorrect
        #confidence threshold for the correction of mistakes by entering the decimal
        #probability after the "autocorrect:" argument.
        autocorrect = False
        autocorrect_confidence = 1
        #If there is at least one instance of a non-directional single "'" or
        #double '"' quote in the "text" string, or if the user has selected the
        #"smart_quotes" or "symmetrical_quotes" options, then all of the directional
        #quotes found within the page are switched to their non-directional counterparts.
        #This would be relevant if the user has trained the CNN model on directional
        #quotes, but didn't get good OCR accuracy when handling them. Alternatively,
        #if the user has trained their model on directional quotes but wants to have
        #symmetrical quotes in the final document, the directional quotes, if present,
        #still need to be changed for the symmetrical quotes. After that first step,
        #if there was at least one instance of a symmetrical quote in the document and
        #that the user didn't specify the "symmetrical_quotes" option, or if they
        #selected the "smart_quotes" option, the appropriate directional quotes will
        #be applied to the page.
        smart_quotes = False
        symmetrical_quotes = False


        if len(sys.argv) > 1:
            #The "try/except" statement will
            #intercept any "ValueErrors" and
            #ask the users to correctly enter
            #the desired values for the variables
            #directly after the colon separating
            #the variable name from the value.
            try:
                for j in range(1, len(sys.argv)):
                    if sys.argv[j][:13] == "top_left_dot:":
                        top_left_dot = True
                        top_left_dot_x = int(sys.argv[j][13:].split(",")[0])
                        top_left_dot_y = int(sys.argv[j][13:].split(",")[1])
                    elif sys.argv[j][:14] == "top_right_dot:":
                        top_right_dot = True
                        top_right_dot_x = int(sys.argv[j][14:].split(",")[0])
                        top_right_dot_y = int(sys.argv[j][14:].split(",")[1])
                    elif sys.argv[j][:12] == "dot_spacing:":
                        inches_between_dots = float(sys.argv[j][12:].strip())
                    elif sys.argv[j][:20] == "dot_diameter_pixels:":
                        dot_diameter_pixels = int(sys.argv[j][20:].strip())
                    elif sys.argv[j][:15] == "dot_line_width:":
                        dot_line_width = int(sys.argv[j][15:].strip())
                    elif sys.argv[j][:10] == "x_overlap:":
                        x_overlap = int(sys.argv[j][10:].strip())
                    elif sys.argv[j][:10] == "y_overlap:":
                        y_overlap = int(sys.argv[j][10:].strip())
                    elif sys.argv[j].lower()[:11] == "top_margin:":
                        inches = float(sys.argv[j][11:].strip())
                        top_margin_y_pixel = round(inches*300)
                    elif sys.argv[j].lower()[:14] == "bottom_margin:":
                        inches = float(sys.argv[j][14:].strip())
                        bottom_margin_y_pixel = 2550-round(inches*300)
                    elif sys.argv[j].lower()[:12] == "left_margin:":
                        inches = float(sys.argv[i][12:].strip())
                        left_margin_x_pixel = round(inches*300)
                    elif sys.argv[j].lower()[:13] == "right_margin:":
                        inches = float(sys.argv[i][13:].strip())
                    elif sys.argv[j].lower()[:14] == "gutter_margin:":
                        gutter_margin_width_pixels = round(float(sys.argv[i].lower()[14:].strip())*300)
                    elif sys.argv[j].lower()[:19] == "lines_between_text:":
                        lines_between_text = int(sys.argv[j].lower()[19:].strip())
                    elif sys.argv[j].lower().split(":")[0] in ["scriptreader", "scriptreader_left", "scriptreader_right"]:
                        #If the user has selected to print some custom
                        #dot grid pages for use in the handwriting OCR
                        #application ScriptReader, they will likely want
                        #to perforate the pages for binding, and so a wider
                        #gutter margins of 0.75 inch is included by default,
                        #which may be overriden if the user has specified
                        #a different gutter margin as the fifth argument.
                        gutter_margin_width_pixels = 0.75*300
                        arguments = sys.argv[j].lower().split(":")[1:]
                        if arguments != [""]:
                            for k in range(len(arguments)):
                                if k == 0:
                                    inches_between_dots = float(arguments[k])
                                elif k == 1:
                                    dot_diameter_pixels = int(arguments[k])
                                elif k == 2:
                                    dot_line_width = int(arguments[k])
                                elif k == 3:
                                    lines_between_text = int(arguments[k])
                                elif k == 4:
                                    gutter_margin_width_pixels = round(float(arguments[k])*300)
                    elif sys.argv[j].lower()[:23] == "basic_autocorrect_lower":
                        basic_autocorrect_lower = True
                    elif sys.argv[j].lower()[:17] == "basic_autocorrect":
                        basic_autocorrect = True
                    elif sys.argv[j].lower()[:11] == "autocorrect":
                        autocorrect = True
                        try:
                            autocorrect_confidence = float(sys.argv[j][12:].strip())
                        except:
                            pass
                    elif sys.argv[j] == "smart_quotes":
                        smart_quotes = True
                    elif sys.argv[j] == "symmetrical_quotes":
                        symmetrical_quotes = True

            except Exception as e:
                print(e)
                print('\nPlease provide the dot spacing in inches (in decimal form and without units) after ' +
                'the "dot_spacing:" argument, along with the number of empty lines in-between lines of text, ' +
                'preceded by "lines_between_text:" Alternatively, simply copy and paste the arguments passed' +
                'in when generating the notebook (excluding the Python call), which can be found in text file ' +
                'entitled "Parameters Passed In.txt", within the "Notebooks" subfolder of your PrintANotebook ' +
                'working folder.')
                print('For example: "dot_spacing: 0.125" "lines_between_text:2"')

        else:
            print('\nPlease provide the dot spacing in inches (in decimal form and without units) after ' +
            'the "dot_spacing:" argument, along with the number of empty lines in-between lines of text, ' +
            'preceded by "lines_between_text:" Alternatively, simply copy and paste the arguments passed' +
            'in when generating the notebook (excluding the Python call), which can be found in text file ' +
            'entitled "Parameters Passed In.txt", within the "Notebooks" subfolder of your PrintANotebook ' +
            'working folder.')
            print('For example: "dot_spacing: 0.125" "lines_between_text:2"')

        #The number of pixels in-between two dots (assuming that the dot grid pages
        #were printed without image resizing) is given using the ratio of 2550 pixels
        #for every 8.5 inches at 300 ppi scan resolution.
        pixels_between_dots = round(inches_between_dots*300)
        #If the user didn't provide a number of pixels for the horizontal overlap of
        #every character segmentation cropped image, it is defaulted to a fourth of
        #the "pixels_between_dots".
        if x_overlap == None:
            x_overlap = round(pixels_between_dots/4)
        #A similar approach is taken for the vertical overlap when performing
        #segmentation, but this time the amount of pixels is proportional to both
        #the number of empty lines in-between every line of text and the number of
        #pixels between every dot.
        if y_overlap == None:
            y_overlap = round(0.75*lines_between_text*pixels_between_dots)

        #The "character_index" will keep track of the index of every
        #character in each of the pages of the dataset, so that every
        #character of a given category has a different file name.
        character_index = 0


        if basic_autocorrect == True or basic_autocorrect_lower == True:
            #The 143K word list ("wlist_match7.txt") was found at the following link
            #(https://www.keithv.com/software/wlist/) and it was assembled by selecting
            #words at the intersection of 7 different word lists, such as the
            #British national corpus.

            #The dictionary text files are retrieved using the "glob" module.
            #If there is more than two text files within the "Dictionary" subfolder
            #of the working folder, an error message is displayed in the Powershell.
            #The same is true if less than two TXT files are found within the folder,
            #as the code needs to read the original dictionary, and its abridged version,
            #without duplicate suggestions bearing only one different letter at a given
            #index. The shorter dictionary file name must also start with "Abridged".
            path_txt = os.path.join(cwd, "Dictionary",  "*.txt")
            txt_files = glob.glob(path_txt)
            if txt_files == []:
                print("\nPlease include a text file (.txt) containing " +
                'the comprehensive list of words that you wish to use in the "Dictionary" subfolder ' +
                'of the working folder, along with its abridged version, of which the file name ' +
                'starts with "Abridged".')
            elif len(txt_files) == 1:
                print("\nPlease include a text file (.txt) containing " +
                'the comprehensive list of words that you wish to use in the "Dictionary" subfolder ' +
                'of the working folder, along with its abridged version, of which the file name ' +
                'starts with "Abridged".')
            elif (len(txt_files) == 2 and os.path.basename(txt_files[0])[:8].lower() == "abridged" or
            os.path.basename(txt_files[1])[:8].lower() == "abridged"):
                for file in txt_files:
                    if os.path.basename(file)[:8].lower() == "abridged":
                        abridged_dict_path = file
                    else:
                        dict_path = file
                        print("dict_path: ", dict_path)
            else:
                print("\nPlease include a text file (.txt) containing " +
                'the comprehensive list of words that you wish to use in the "Dictionary" subfolder ' +
                'of the working folder, along with its abridged version, of which the file name ' +
                'starts with "Abridged".')


            #The 143 word list ("wlist_match7.txt") was found at the following link
            #(https://www.keithv.com/software/wlist/) and it was assembled by selecting
            #words at the intersection of 7 different word lists, such as the
            #British national corpus.
            with open(dict_path, "r", encoding="utf-8") as dict:
                english_dict = dict.readlines()
            #The dictionary "Abridged wlist_match12.txt" comprises the words derived from
            #the more restricted dictionary 27K-word dictionary ("wlist_match12.txt"),
            #containing the most common words in the English language that are at least
            #four letter long as well as unambiguous, in that they differ from one another
            #by more than one letter/sequence of letters.
            #Of note, as the code doesn't know where the OCR error has occured within the word,
            #it is possible that that it will select a word that isn't the right one from the
            #shortened word list.

            #If the user has selected one of the  "basic_autocorrect" options, then the code
            #will screen the misspelled words in the page (words that aren't found
            #within the more extensive "english_dict", here based on a 143K-word
            #dictionary "wlist_match7.txt") to see if any of them could be corrected by
            #altering one letter, and that the resulting word could be found in
            #the 15K-unambiguous word dictionary "short_english_dict", itself
            #derived from the 27K-word dictionary "wlist_match12.txt". If so,
            #the substitution would be made.
            with open(abridged_dict_path, "r", encoding="utf-8") as short_dict:
                short_english_dict = short_dict.readlines()

        if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
            with open(os.path.join(path, OCR_text_file_name + '-OCR (autocorrect).rtf'), 'a+', encoding="utf-8") as e:
                e.write(r"{\rtf1 \ansi \deff0 {\fonttbl {\f0 Ubuntu;}} \f0 \fs24 \par ")


        with open(os.path.join(path, OCR_text_file_name + '-OCR.rtf'), 'a+', encoding="utf-8") as f:
            f.write(r"{\rtf1 \ansi \deff0 {\fonttbl {\f0 Ubuntu;}} \f0 \fs24 \par ")
            #This code obtains the individual character coordinates from the image files
            #listed in the "JPEG_file_names" list and generates JPEG images with overlaid
            #character rectangles, named after the original files, but with the added
            #"with character rectangles" suffix.

            for i in range(len(JPEG_file_names)):
                print(f'\nCurrently processing image entitled: "{JPEG_file_names[i]}"\n')
                #The "image_top_margin_y_pixel" and
                #"image_bottom_margin_y_pixel" variables
                #are initialized as the starting values
                #of "top_margin_y_pixel" and
                #"bottom_margin_y_pixel", respectively,
                #as these values may be altered when the
                #whole dot grid is brought down. This way,
                #the original values are maintained for
                #the next images.
                image_top_margin_y_pixel = top_margin_y_pixel
                image_bottom_margin_y_pixel = bottom_margin_y_pixel

                #The image is loaded using the "cv2.imread" method.
                text_image = cv2.imread(os.path.join(cwd, "OCR Raw Data", str(JPEG_file_names[i])))
                #A copy of the image is made in order to add the character rectangles to it.
                text_image_copy = text_image.copy()
                #The width of the image is determined. This will be useful when determining
                #the "x" coordinate where the start of the page is located.
                imgheight=text_image_copy.shape[0]
                imgwidth=text_image_copy.shape[1]
                #Convert image from RGB to grayscale (This will be used in the actual
                #character cropping, as grayscaleimages will be used to train the model
                #and get OCR predictions. This way, the user could use different colored
                #pens and the model should still work nicely).
                text_image_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)

                #The "get_dot_x_coordinates(inches_between_dots, dot_diameter_pixels)"
                #function populates the "dot_x_coordinates" list with the same row "x"
                #coordinates, distance in-between dots ("inches_between_dots") as well as
                #left and  gutter margins, ("left_margin_x_pixel" and "gutter_margin_width_pixels",
                #respectively) that were used to generate the actual dot grid notebook pages.
                def get_dot_x_coordinates(inches_between_dots, dot_diameter_pixels):
                    #**IMPORTANT!!** You cannot start your actual JPEG file names
                    #with "back", as the presence of the "back" prefix in the file name
                    #designetes whether the page is even or odd numbered, with the
                    #file names for even-numbered pages starting with "back".

                    #The pixel at which the actual page scan begins is dermined by
                    #taking dividing the difference between the image width and
                    #1650, which is the number of pixels for the page width of 5.5
                    #inches at 300 ppi resolution, by two.
                    image_border_to_start_of_page_x_pixels = round(imgwidth-1650)/2
                    #If the page is even-numbered (the file name is starts with "back"
                    #for "back page"), then the "starting_x" pixel is initialized to
                    #"left_margin_x_pixel".
                    if JPEG_file_names[i][:4].lower() == "back":
                        starting_x = left_margin_x_pixel + image_border_to_start_of_page_x_pixels
                        pixel_increment =  pixels_between_dots
                        dot_x_coordinates = []
                        #while "starting_x" is within range of the gutter margin
                        #pixel on the page (imgwidth-image_border_to_start_of_page_x_pixels)
                        #-gutter_margin_width_pixels), it will be added to the list
                        #"dot_x_coordinates" and the "starting_x" will be incremented
                        #by the number of pixels in-between dots.
                        while starting_x <= (imgwidth-image_border_to_start_of_page_x_pixels)-gutter_margin_width_pixels:
                            dot_x_coordinates.append(starting_x)
                            starting_x += pixel_increment
                        return dot_x_coordinates
                    #If the page is odd-numbered (right hand page, so the front side),
                    #the file name doesn't start with "back". The "starting_x" pixel
                    #is initialized as "right_margin_x_pixel". Mirroring the above "if"
                    #statement, while the "starting_x" (initialized to the "x" pixel
                    #of the right margin) is over the gutter margin (now on the left
                    #side of the page), it is included in the "dot_x_coordinates" list.
                    #the sorted list is returned, as the dots are added to the list
                    #from the right to the left side of the page.
                    else:
                        starting_x = right_margin_x_pixel + image_border_to_start_of_page_x_pixels
                        pixel_increment =  pixels_between_dots
                        dot_x_coordinates = []
                        while starting_x >= image_border_to_start_of_page_x_pixels + gutter_margin_width_pixels:
                            dot_x_coordinates.append(starting_x)
                            starting_x -= pixel_increment
                        return sorted(dot_x_coordinates)

                #The "get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels)"
                #function populates the "dot_y_coordinates" list with the same line "y"
                #coordinates, distance in-between dots ("inches_between_dots") as well as
                #top and bottom margins ("image_top_margin_y_pixel" and "image_bottom_margin_y_pixel",
                #respectively) that were used to generate the actual dot grid notebook pages.
                #Starting from the top margin, "y" coordinates will be added to the "dot_y_coordinates"
                #list in increments of the pixel distance in-between dots ("pixels_between_dots").
                def get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels):
                    starting_y = image_top_margin_y_pixel
                    pixel_increment = pixels_between_dots
                    dot_y_coordinates = []
                    while starting_y <= image_bottom_margin_y_pixel:
                        dot_y_coordinates.append(starting_y)
                        starting_y += pixel_increment
                    return dot_y_coordinates

                #The row "x" and line "y" coordinates of the dot grid are gathered
                #by calling the "get_dot_x_coordinates" and "get_dot_y_coordinates",
                #respectively.
                dot_x_coordinates = get_dot_x_coordinates(inches_between_dots, dot_diameter_pixels)
                dot_y_coordinates = get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels)

                #The list of line indices where characters will be segmented ("text_line_numbers")
                #is initialized including the zero index, as the first line of text needs to be
                #on the first line, and then at a regular interval thereafter after that. There
                #is a default of three empty lines in-between every line of text, to minimize
                #the overlapping of ascenders and descenders of adjacent text lines.
                text_line_numbers = [0]
                #Here there is one less dot than the total number of lines, so there is no
                #need to add "+1" after "len(dot_y_coordinates)"
                for j in range(len(dot_y_coordinates)):
                    #If the current "dot_y_coordinates" list index is prior to the penultimate
                    #list index (as room needs to be provided to add a "y" coordinate, and
                    #if the current index is equal to that of the last text line, plus the number
                    #of empty lines in-between text lines plus 1 (to account for the fact that this
                    #version of the code does not include the bottom horizontal line of dots for each
                    #line of text), then it is included in the list of text line indices "text_line_numbers".
                    if j < len(dot_y_coordinates)-2 and j == text_line_numbers[-1] + lines_between_text + 1:
                        text_line_numbers.append(j)

                #If the lower "y" coordinate of the last set of two successive horizontal dot lines framing
                #a text line, plus the pixel diameter of a dot, plus the vertical overlap allocated to
                #accomodate for ascenders and descenders when handwriting (round(0.40*lines_between_text*
                #inches_between_dots*300)) is inferior to the lower margin of the page ("image_bottom_margin_y_pixel"),
                #it means that there is likely to be excessive space at the bottom of the page, relative to the space
                #above the header. To improve the page layout esthetics, the whole page will be shifted down by the
                #difference in pixels in-between the lower margin of the page and point described above, by adjusting
                #the margins accordingly. All of the "y_coordinate" lists therefore need to be recalculated at this
                #point, to reflect the changes in margins. As opposed to the PrintANotebook code, "+1" needs to be
                #added to "text_line_numbers[-1]", as only the first dot line index of every text line is included
                #in "text_line_numbers" in this version.
                top_y_shift = 0
                if (dot_y_coordinates[text_line_numbers[-1]+1] + dot_diameter_pixels +
                round(0.40*lines_between_text*inches_between_dots*300) < image_bottom_margin_y_pixel):
                    top_y_shift = (image_bottom_margin_y_pixel-dot_y_coordinates[text_line_numbers[-1]+1] -
                    (dot_diameter_pixels + round(0.40*lines_between_text*inches_between_dots*300)))
                    image_top_margin_y_pixel += top_y_shift
                    image_bottom_margin_y_pixel -+ (image_bottom_margin_y_pixel-dot_y_coordinates[text_line_numbers[-1]+1] +
                    (dot_diameter_pixels + round(0.40*lines_between_text*inches_between_dots*300)))
                    dot_y_coordinates = get_dot_y_coordinates(inches_between_dots, dot_diameter_pixels)


                #The image's numpy array is filtered using the np.where() function to convert pixels
                #lighter than 200 on the grayscale scale to 0 and darker pixels to 1. The rows are added up
                #(summation along the 1 axis) to determine how many non-white pixels there are for a given
                #y coordinate. The same is done for the columns, with a summation along the 0 axis.
                #image_filtered = np.where(text_image_gray>200, 0, 1)
                image_filtered = np.where(text_image_gray>100, 0, 1)
                y_pixels_left_square = np.sum(image_filtered[:round(dot_y_coordinates[0]-100),
                round(dot_x_coordinates[0]-100):round(dot_x_coordinates[0]+150)], axis=1)
                x_pixels_left_square = np.sum(image_filtered[:round(dot_y_coordinates[0]-100),
                round(dot_x_coordinates[0]-100):round(dot_x_coordinates[0]+150)], axis=0)

                #Only the "y" pixels where there are more than 10 "x" pixels under a grayscale value of 200 are
                #retained in "y_pixels_left_square". The difference between the index of the first and last "y"
                #pixels meeting these requirements will give the height of the square:
                #(y_pixels_left_square[-1]-y_pixels_left_square[0])

                y_pixels_left_square = np.where(y_pixels_left_square > 10)[0]

                #The center of the square on the "y" axis may be reached by adding the vertical distance
                #from the top of the image to the topmost horizontal side of the square ("y_pixels_left_square[0]",
                #as the slicing used in the "np.num()" operation started from the top of the image) to the half-height
                #of the square (y_pixels_left_square[-1]-y_pixels_left_square[0])/2).
                y_center_left_square = round(y_pixels_left_square[0] + (y_pixels_left_square[-1]-y_pixels_left_square[0])/2)

                x_pixels_left_square = np.where(x_pixels_left_square > 10)[0]

                #In order to reach the center of the square on the "x" axis, we need to add the amount of pixels needed
                #to reach the leftmost coordinate of the "image_filtered" slicing used in the "np.sum()" operation
                #(dot_x_coordinates[0]-100), then add the amount of pixels within that slice to reach
                #the leftmost vertical corner of the square (x_pixels_left_square[0]), and finally add the half-width of
                #the square (x_pixels_left_square[-1]-x_pixels_left_square[0])/2).
                x_center_left_square = round(dot_x_coordinates[0]-100 + x_pixels_left_square[0] +
                (x_pixels_left_square[-1]-x_pixels_left_square[0])/2)

                #The equivalent code to the one above is used for the gutter margin square on even (left-hand) pages.
                y_pixels_right_square = np.sum(image_filtered[:round(dot_y_coordinates[0]-100),
                round(dot_x_coordinates[-1]-150): round(dot_x_coordinates[-1]+100)], axis=1)
                x_pixels_right_square = np.sum(image_filtered[:round(dot_y_coordinates[0]-100),
                round(dot_x_coordinates[-1]-150): round(dot_x_coordinates[-1]+100)], axis=0)

                y_pixels_right_square = np.where(y_pixels_right_square > 10)[0]

                y_center_right_square = round(y_pixels_right_square[0] +
                (y_pixels_right_square[-1]-y_pixels_right_square[0])/2)

                x_pixels_right_square = np.where(x_pixels_right_square > 10)[0]

                #In order to reach the center of the square on the "x" axis, we need to add the amount
                #of pixels needed to reach the leftmost coordinate of the "image_filtered" slicing used
                #in the "np.sum()" operation ("dot_x_coordinates[-1]-150"), then add the amount of pixels
                #within that slice to reach the leftmost vertical corner of the square
                #("x_pixels_right_square[0]"), and finally add the half-width of
                #the square (x_pixels_right_square[-1]-x_pixels_right_square[0])/2).
                x_center_right_square = (round(dot_x_coordinates[-1]-150 +
                x_pixels_right_square[0] + (x_pixels_right_square[-1]-x_pixels_right_square[0])/2))

                #The slope of the line connecting center of the two corner squares is calculated and will
                #be used to determine the angle used in trigonometric calculations using the
                #measurements of the untilted dot grid as the hypothenuse, in order to correct for
                #the image rotation, assuming that all pages in the stack of pages will be tilted
                #in a similar way.
                slope = ((y_center_right_square - y_center_left_square)/
                (x_center_right_square-x_center_left_square))
                slope_angle = np.arctan(slope)

                #As the black rectangles are shifted down by the same amount of pixels as the dot grid,
                #the number of pixels on the "y" axis, between the center of the squares and the topmost
                #dots remains constant and can be used to determine the exact "y" coordinates of the top
                #left and top right dots, based on the "y" center coordinate of the left and right squares,
                #respectively.
                pixels_between_centers_of_black_square_and_top_dot = top_margin_y_pixel - (left_margin_x_pixel + 25)
                top_left_dot_y = round(y_center_left_square + pixels_between_centers_of_black_square_and_top_dot)
                top_right_dot_y = round(y_center_right_square + pixels_between_centers_of_black_square_and_top_dot)

                #The x coordinates of the top left and right dots are determined by
                #trigonometric calculations, using the known values for the margins
                #and horizontal distance between these dots in the untilted image
                #as the hypothenuses, and the tilt angle.
                top_left_dot_x = round(x_center_left_square - 25*np.cos(slope_angle))
                top_right_dot_x = round(x_center_left_square - 25*np.cos(slope_angle) +
                (dot_x_coordinates[-1]-dot_x_coordinates[0])*np.cos(slope_angle))


                #The rectangles are drawn on "text_image_copy" to allow users to evaluate
                #how well the segmentation has proceeded.

                #Left black square: A red rectangle is drawn so as to outline the
                #black square
                cv2.rectangle(text_image_copy, (round(x_center_left_square-25*np.cos(slope_angle)),
                round(y_center_left_square-25*np.cos(slope_angle))), (round(x_center_left_square+25*np.cos(slope_angle)),
                round(y_center_left_square+25*np.cos(slope_angle))), (0,0,255),3)
                #Left black square: A blue rectangle is drawn so as to outline the
                #slicing region of the original "image_filtered" for the "np.sum" operation
                cv2.rectangle(text_image_copy, (round(dot_x_coordinates[0]-100),
                0), (round(dot_x_coordinates[0]+150),
                round(dot_y_coordinates[0]-100)), (255,0,0),3)

                #Right black square: A red rectangle is drawn so as to outline the
                #black square
                cv2.rectangle(text_image_copy, (round(x_center_right_square-25*np.cos(slope_angle)),
                round(y_center_right_square-25*np.cos(slope_angle))), (round(x_center_right_square+25*np.cos(slope_angle)),
                round(y_center_right_square+25*np.cos(slope_angle))), (0,0,255),3)
                #Right black square: A blue rectangle is drawn so as to outline the
                #slicing region of the original "image_filtered" for the "np.sum" operation
                cv2.rectangle(text_image_copy, (round(dot_x_coordinates[-1]-150),
                0), (round(dot_x_coordinates[-1]+100),
                round(dot_y_coordinates[0]-100)), (255,0,0),3)

                #The hypothenuse here is the horizontal dimension between the next character's "x" coordinate and
                #the starting "x" on the text line, in the untilted dot grid. So the next "x" coordinate is determined
                #by multiplying that pixel count by the cosine of the slope angle, and then adding the current "x"
                #coordinate, to allow the segmentation to walk forward on the line.
                def get_next_x(k):
                    next_x = round(current_x + (dot_x_coordinates[k+1]-dot_x_coordinates[0])*np.cos(slope_angle) -
                    (dot_x_coordinates[k]-dot_x_coordinates[0])*np.cos(slope_angle))
                    return next_x

                #The hypothenuse here is the horizontal dimension between the next character's "x" coordinate and
                #the starting "x" on the text line, in the untilted dot grid. So the next "y" coordinate is determined
                #by multiplying that pixel count by the sine of the slope angle, and then adding the "y" coordinate
                #of the starting top dot of that text line.
                def get_next_y(k):
                    next_y = round(next_line_y + ((dot_x_coordinates[k+1]-dot_x_coordinates[0])*np.sin(slope_angle)))
                    return next_y

                #The list of character "x,y" coordinates is populated with the character coordinates of each
                #of the line indices within the "text_line_numbers" list. Each character list of coordinates
                #is comprised of the top left "x,y" coordinates sublist, followed by another sublist of "x,y"
                #coordinates of the bottom right corner of the character rectangle ("[[top_left_x, "top_left_y"],
                #[bottom_right_x, bottom_right_y]]"). In order to determine the "x,y" coordinates of the lower
                #right corner, trigonometric calculations must be performed using the tilt angle of the scanned
                #page.
                chars_x_y_coordinates = []
                for j in range(len(text_line_numbers)):
                    #The "if" statement below will be discussed in more detail, and covers the situation where
                    #the slope is positive ("if" statement), meaning that the page is tilted clockwise,
                    #as "y" coordinates increase as we go down the image. The "next_line_x" is calculated by
                    #making the difference between the starting "y" coordinate of the next line and that of the
                    #first line in the untilted page (which gives the cumulative vertical distance from the origin up
                    #to che current line in the untilted page), after being multiplied by the sine of the scanned page
                    #tilt angle. This result is then subtracted from the "top_left_dot_x". A similar calculation is
                    #performed to determine "next_line_y", but this time with the cosine of the slope angle.
                    if slope > 0:
                        next_line_x = (round(top_left_dot_x - (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.sin(slope_angle)))
                        next_line_y = (round(top_left_dot_y + (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.cos(slope_angle)))
                        #As the page is tilted clockwise, its rightmost point should be the top right corner.
                        #The "x" coordinate of the top right corner will then act as the "x_threshold", beyond
                        #which no chracter is to be segmented. The horizontal pixels that extend outside of the
                        #four dot square chracter grid ("x_overlap") is added to allow for the last character on
                        #the line to have a horizontal overlap as well.
                        x_threshold = top_right_dot_x + x_overlap
                    elif slope < 0:
                        next_line_x = (round(top_left_dot_x - (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.sin(slope_angle)))
                        next_line_y = (round(top_left_dot_y + (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.cos(slope_angle)))
                        #If the slope is negative (the image is tilted counter-clockwise, given that y coordinates
                        #increase going down, then the maximal "x" coordinate will be greater than "top_right_dot_x",
                        #so the horizontal distance is determined using a right angle triangle with a hypothenuse
                        #equal to the total height of the dot grid in the untilted page).
                        x_threshold = (round(top_right_dot_x - (dot_y_coordinates[text_line_numbers[-1]]-
                        dot_y_coordinates[text_line_numbers[0]])*np.sin(slope_angle) + x_overlap))
                    #If the scanned page is untilted (well done!), the "next_line_x" would line up exactly with the
                    #"top_left_dot" the "next_line_y" would be the the cumulative vertical distance from the origin
                    #up to che current line in the untilted page, which doesn't need to be corrected by trigonometric
                    #calculations in this case, and adding this result to the top left corner "y" coordinate.
                    elif slope == 0:
                        next_line_x = top_left_dot_x
                        next_line_y = (round(top_left_dot_y + (dot_y_coordinates[text_line_numbers[j]]-
                        dot_y_coordinates[text_line_numbers[0]])))
                        #As the page isn't tilted, all of the right coordinates will line up to the top right dot
                        #"x" coordinate, with the addition of the "x_overlap".
                        x_threshold = top_right_dot_x + x_overlap
                    #A new empty list is added to the "chars_x_y_coordinates" list at the start of every new line.
                    chars_x_y_coordinates.append([])

                    #The "for" loop below loops through every "x" coordinate within the line in order to gather the
                    #"x,y" coordinates of every segmented character on the text lines.
                    for k in range(len(dot_x_coordinates)-1):
                        #The first character of every line has its "x,y" coordinates initialized
                        #to the "next_line_x" and "next_line_y", respectively, determined above.
                        if k == 0:
                            current_x = next_line_x
                            current_y = next_line_y
                        #The characters following the first character on every line get assigned a value
                        #of the "next_x" and "next_y" determined in the previous iteration of the loop.
                        else:
                            current_x = next_x
                            current_y = next_y
                        #The new "next_x" and "next_y" values are determined using the "get_next_x(current_x,k)"
                        #and "get_next_y(current_y,k)" functions, respectively.
                        next_x = get_next_x(k)
                        next_y = get_next_y(k)
                        #If the "next_x" is lower than the "x_threshold", then the rectangle is
                        #included in the "chars_x_y_coordinates" list at the current "j" line index.
                        if next_x < x_threshold:
                            chars_x_y_coordinates[j].append([[current_x,
                            current_y], [next_x, next_y+pixels_between_dots]])
                            #The rectangles are drawn on "text_image_copy" to allow users to evaluate
                            #how well the segmentation has proceeded.
                            (cv2.rectangle(text_image_copy, (chars_x_y_coordinates[j][k][0][0],
                            chars_x_y_coordinates[j][k][0][1]), (chars_x_y_coordinates[j][k][1][0],
                            chars_x_y_coordinates[j][k][1][1]), (0,255,0),3))

                #If there is an empty line at the end of the "chars_x_y_coordinates"
                #it is sliced out.
                if chars_x_y_coordinates[-1] == []:
                    chars_x_y_coordinates = chars_x_y_coordinates[:-1]

                #The list "chars_x_y_coordinates" is screened
                #to find the lowest and highest vertical ("x" axis)
                #horizontal ("y" axis) dimensions of the character
                #rectangles within the list. This is important because
                #it will allow to ensure that all cropped character
                #images are of the same vertical and horizontal
                #dimensions, which is essential for the OCR step.
                largest_x_dimension = pixels_between_dots
                smallest_x_dimension = pixels_between_dots
                largest_y_dimension = pixels_between_dots
                smallest_y_dimension = pixels_between_dots
                for line in chars_x_y_coordinates:
                    for char in line:
                        x_dimension = char[1][0] - char[0][0]
                        if x_dimension > largest_x_dimension:
                            largest_x_dimension = x_dimension
                        if x_dimension < smallest_x_dimension:
                            smallest_x_dimension = x_dimension
                        y_dimension = char[1][1] - char[0][1]
                        if y_dimension > largest_y_dimension:
                            largest_y_dimension = y_dimension
                        if y_dimension < smallest_y_dimension:
                            smallest_y_dimension = y_dimension

                #The "x_overlap" and "y_overlap" are automatically adjusted
                #in order to accomodate instances where the difference in-between
                #the maximal and minimal vertical and horizontal measurements are
                #above twice the value of "x_overlap" and "y_overlap". With these
                #adjustments, each character will be investigated in the "for" loop
                #below in order to adjust the overlap so as to ensure that every
                #rectangle has exactly the same dimensions, which is important
                #for the OCR step.
                if largest_x_dimension-smallest_y_dimension > 2*x_overlap:
                    x_overlap = round(largest_x_dimension-smallest_y_dimension)
                if largest_y_dimension-smallest_y_dimension > 2*y_overlap:
                    y_overlap = round(largest_y_dimension-smallest_y_dimension)

                #The "for" loop below cycles through every character segmentation "x,y" coordinates
                #to check whether the horizontal or vertical measurements are above those of uncorrected
                #cropped squares having a dimension equal to "pixels_between_dots". The adjustments are
                #made accordingly to "x_overlap" and "y_overlap" to ensure that the final segmentation
                #rectangles are all of the same dimensions, which is important for the OCR step.
                for j in range(len(chars_x_y_coordinates)):
                    for k in range(len(chars_x_y_coordinates[j])):
                        x_dimension = chars_x_y_coordinates[j][k][1][0] - chars_x_y_coordinates[j][k][0][0]
                        y_dimension = chars_x_y_coordinates[j][k][1][1] - chars_x_y_coordinates[j][k][0][1]
                        #If "x_dimension" is greater than the size of an uncorrected cropped square having a
                        #dimension of "pixels_between_dots", then some pixels need to be subtracted from
                        #"custom_x_overlap" in order to ensure that the cropped rectangle is of the same
                        #size as the others. If the difference between the "x_dimension" and "pixels_between_dots"
                        #is an odd number, it means that a different amount of pixels needs to be subtracted
                        #from the "x_overlap" on the left and right sides.
                        if x_dimension > pixels_between_dots and (x_dimension-pixels_between_dots)%2 != 0:
                            #The floor division is used to arbitrarily subtract the rounded down pixel number
                            #from the left side, while the "math.ceil" method is called upon to round up the
                            #number of pixels that will be subtracted from the right side. This way, the correct
                            #amount of pixels will be removed on either side in order for the cropped rectangle
                            #to have the same dimensions as the others.
                            custom_x_overlap_left = -(x_overlap - ((x_dimension-pixels_between_dots)//2))
                            custom_x_overlap_right = (x_overlap - math.ceil((x_dimension-pixels_between_dots)/2))
                        #If the diffrence is even, then the same number of pixels will be removed on both
                        #sides of the character rectangle.
                        elif x_dimension > pixels_between_dots and (x_dimension-pixels_between_dots)%2 == 0:
                            custom_x_overlap_left = -(x_overlap - int((x_dimension-pixels_between_dots)/2))
                            custom_x_overlap_right = (x_overlap - int((x_dimension-pixels_between_dots)/2))
                        #If the "x_dimension" is lower than "pixels_between_dots", then some pixels need
                        #to be added to "x_overlap" in order for the resulting cropped rectangle to be
                        #of the same dimensions as the others.
                        elif x_dimension < pixels_between_dots and (pixels_between_dots-x_dimension)%2 != 0:
                            custom_x_overlap_left = -(x_overlap + ((pixels_between_dots-x_dimension)//2))
                            custom_x_overlap_right = (x_overlap + math.ceil((pixels_between_dots-x_dimension)/2))

                        elif x_dimension < pixels_between_dots and (pixels_between_dots-x_dimension)%2 == 0:
                            custom_x_overlap_left = -(x_overlap + int((pixels_between_dots-x_dimension)/2))
                            custom_x_overlap_right = (x_overlap + int((pixels_between_dots-x_dimension)/2))
                        #If the "x_dimension" is equal to "pixels_between_dots", then no adjustments
                        #need to be made to "x_overlap".
                        elif x_dimension == pixels_between_dots:
                            custom_x_overlap_left = -x_overlap
                            custom_x_overlap_right = x_overlap
                        #The same logic is used in the "y" dimension.
                        if y_dimension > pixels_between_dots and (y_dimension-pixels_between_dots)%2 != 0:
                            custom_y_overlap_top = -(y_overlap - ((y_dimension-pixels_between_dots)//2))
                            custom_y_overlap_bottom = (y_overlap - math.ceil((y_dimension-pixels_between_dots)/2))
                        elif y_dimension > pixels_between_dots and (y_dimension-pixels_between_dots)%2 == 0:
                            custom_y_overlap_top = -(y_overlap - int((y_dimension-pixels_between_dots)/2))
                            custom_y_overlap_bottom = (y_overlap - int((y_dimension-pixels_between_dots)/2))
                        elif y_dimension < pixels_between_dots and (pixels_between_dots-y_dimension)%2 != 0:
                            custom_y_overlap_top = -(y_overlap + ((pixels_between_dots-y_dimension)//2))
                            custom_y_overlap_bottom = (y_overlap + math.ceil((pixels_between_dots-y_dimension)/2))
                        elif y_dimension < pixels_between_dots and (pixels_between_dots-y_dimension)%2 == 0:
                            custom_y_overlap_top = -(y_overlap + int((pixels_between_dots-y_dimension)/2))
                            custom_y_overlap_bottom = (y_overlap + int((pixels_between_dots-y_dimension)/2))
                        elif y_dimension == pixels_between_dots:
                            custom_y_overlap_top = -y_overlap
                            custom_y_overlap_bottom = y_overlap

                        #The character "x,y" coordinates within the "chars_x_y_coordinates" list are
                        #updated to reflect the custom "x" and "y" overlaps determined above.
                        chars_x_y_coordinates[j][k] = [[chars_x_y_coordinates[j][k][0][0] +
                        custom_x_overlap_left, chars_x_y_coordinates[j][k][0][1] + custom_y_overlap_top],
                        [chars_x_y_coordinates[j][k][1][0] + custom_x_overlap_right,
                        chars_x_y_coordinates[j][k][1][1] + custom_y_overlap_bottom]]

                if not os.path.exists(os.path.join(cwd, "Page image files with rectangles")):
                    os.makedirs(os.path.join(cwd, "Page image files with rectangles"))
                (cv2.imwrite(os.path.join(cwd, 'Page image files with rectangles', JPEG_file_names[i][:-4] +
                ' with character rectangles.jpg'), text_image_copy))

                char_files = []
                char_index = 0
                for j in range(len(chars_x_y_coordinates)):
                    for k in range(len(chars_x_y_coordinates[j])):
                        (cv2.imwrite(os.path.join(path, str(char_index) + ".jpg"),
                        text_image_gray[chars_x_y_coordinates[j][k][0][1]:
                        chars_x_y_coordinates[j][k][1][1], chars_x_y_coordinates[j][k][0][0]:
                        chars_x_y_coordinates[j][k][1][0]]))
                        char_files.append(os.path.join(path, str(char_index) + ".jpg"))
                        char_index += 1

                #Generate batch of individual character ".jpg" images to be submitted
                #to the model for prediction.
                data_block = DataBlock(
                                blocks = (ImageBlock, CategoryBlock),
                                get_items = get_image_files, batch_tfms = Normalize()
                                )
                dls = data_block.dataloaders(path, bs=64)
                dl = learn.dls.test_dl(char_files, shuffle=False)
                #Obtain softmax results in the form of a one-hot vector per character
                preds = learn.get_preds(dl=dl)[0].softmax(dim=1)
                #Determine which is the category index for the argmax of the character one-hot vectors.
                preds_argmax = preds.argmax(dim=1).tolist()
                #Convert the category index for each character to its label and assemble
                #a list of labels by list comprehension.
                text = [learn.dls.vocab[preds_argmax[j]] for j in range(len(preds_argmax))]

                #If you want to print out the dictionary mapping the labels to the label
                #indices, uncomment the following line:
                # print(learn.dls.vocab.o2i)

                #Once the "text" list of predicted characters has been populated, delete the individual
                #character ".jpg" images used for OCR (you can comment out the following lines of
                #code should you want to retain them for troubleshooting purposes).
                for j in range(len(text)):
                    os.remove(os.path.join(path, str(j) + '.jpg'))

                #Substitute the actual character labels for the labels that were written in long
                #form for compatibility reasons.
                for j in range(len(text)-1, -1, -1):
                    #If the label is "empty" (a typo overlaid with a hashtag symbol),
                    #replace those characters with " ". Superfluous spaces are removed at the end of the code
                    #(text = "".join(text).replace("  ", " ")).
                    if text[j] == "empty":
                        text[j] = ""
                    #If the label is a "space", it is replaced with " ".
                    elif text[j] == "space":
                        text[j] = " "
                    #If the label is "forward slash", it is replaced with a forward slash.
                    elif text[j] == "forward slash":
                        text[j] = "/"
                    #If the label is "backslash", it is replaced with a backslash.
                    elif text[j] == "backslash":
                        text[j] = "\\"
                    #If the label is "pipe", it is replaced with a pipe symbol.
                    elif text[j] == "pipe":
                        text[j] = "|"
                    #If the label is "dollar sign", it is replaced for a dollar sign.
                    elif text[j] == "dollar sign":
                        text[j] = "$"
                    #If the label is "plus sign", it is replaced by a plus sign.
                    elif text[j] == "plus sign":
                        text[j] = "+"
                    #If the label is "equals sign", it is replaced by an equals sign.
                    elif text[j] == "equals sign":
                        text[j] = "="
                    #If the label is "question mark", it is replaced by a question mark.
                    elif text[j] == "question mark":
                        text[j] = "?"
                    #If the label is "exclamation mark", it is replaced by an exclamation mark.
                    elif text[j] == "exclamation mark":
                        text[j] = "!"
                    #If the label is "period", it is replaced with ".".
                    elif text[j] == "period":
                        text[j] = "."
                    #If the label is "colon", it is replaced by a colon.
                    elif text[j] == "colon":
                        text[j] = ":"
                    #If the label is "at sign", it is replaced by an at sign.
                    elif text[j] == "at sign":
                        text[j] = "@"
                    #If the label is "grave accent", it is replaced by a grave accent.
                    elif text[j] == "grave accent":
                        text[j] = "`"
                    #If the label is "single quote", it is replaced by a symmetrical single
                    #quote ("'"), which will later be converted to the directional quote.
                    elif text[j] == "single quote":
                        text[j] = "'"
                    #If the label is "double quote", it is replaced by a symmetrical double
                    #quote ('"'), which will later be converted to the directional quote.
                    elif text[j] == "double quote":
                        text[j] = '"'
                    #If the label is "hashtag", it is replaced by a hashtag.
                    elif text[j] == "hashtag":
                        text[j] = "#"
                    #If the label is "lesser-than sign", it is replaced by a lesser-than sign.
                    elif text[j] == "lesser-than sign":
                        text[j] = "<"
                    #If the label is "greater-than sign", it is replaced by a greater-than sign.
                    elif text[j] == "greater-than sign":
                        text[j] = ">"
                    #If the label is "asterisk", it is replaced by an asterisk.
                    elif text[j] == "asterisk":
                        text[j] = "*"
                    #If the label is "percent", it is replaced by a percent symbol.
                    elif text[j] == "percent":
                        text[j] = "%"
                    #If the label is "ampersand", it is replaced with an ampersand symbol.
                    elif text[j] == "ampersand":
                        text[j] = "&"
                    #If the label is "left curly bracket", it is changed for a left curly bracket.
                    elif text[j] == "left curly bracket":
                        text[j] = "{"
                    #If the label is "right curly bracket", it is changed for a right curly bracket.
                    elif text[j] == "right curly bracket":
                        text[j] = "}"

                #The following "for" loop removes spaces before and after hyphens, to
                #ensure that hyphenated words do not include spaces.
                for j in range(1, len(text)-1):
                    if text[j] in ["–","—","—", "-"] and text[j-1].strip() == "":
                        text[j-1] = ""
                    elif text[j] in ["–","—","—", "-"] and text[j+1].strip() == "":
                        text[j+1] = ""

                '''DEFAULT (BASIC) RTF FORMATTING MODE'''
                #Join the elements of the "text" list and perform a series of string substitutions
                #to yield the OCR text. The following line replaces replaces "\par" with "\par\pard\tab ",
                #to set the new paragraph's formatting to the default settings and to include a tab at
                #the start of every new paragraph. After that, any new paragraphs that had centered \
                #alignment ("\qc") have their tab removed, so that the centered alignment isn't shifted
                #through the inclusion of a superfluous tab. For full rtf functionalities (without this
                #last simplification), comment the next line out and activate the lines after it instead.
                #An extra space is included after every RTF command, in case the user forgot to include
                #the optional space after RTF commands, which would "eat up" a regular space. Any spaces
                #in excess of two are automatically filtered out later in the code (after changing the RTF
                #escapes). Finally, superfluous spaces before punctuation marks or closing brackets are removed.
                text = ("".join(text).replace(r"\par", r"\par\pard\tab").replace(r"\par\pard\tab \qc", r"\par\pard\qc ")
                .replace(r"\par\pard\tab\qc", r"\par\pard\qc ").replace(r"\b0", r"\b0 ").replace(r"\i0", r"\i0 ")
                .replace(r"\scaps0", r"\scaps0 ").replace(r"\strike0", r"\strike0 ").replace(r"\ul0", r"\ul0 ")
                .replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" ;", ";").replace("( ", "(")
                .replace(" )", ")").replace(" ?", "?").replace(" !", "!"))


                '''ADVANCED RTF FORMATTING MODE'''
                #See comments above in the Default (Basic) RTF formatting mode for more details.
                # text = ("".join(text).replace(r"\b0", r"\b0 ").replace(r"\i0", r"\i0 ")
                # .replace(r"\scaps0", r"\scaps0 ").replace(r"\strike0", r"\strike0 ")
                # .replace(r"\ul0", r"\ul0 ").replace(" .", ".").replace(" ,", ",")
                # .replace(" :", ":").replace(" ;", ";").replace("( ", "(")
                # .replace(" )", ")").replace(" ?", "?").replace(" !", "!"))


                #If the user has selected the "basic_autocorrect_lower" option, then the code
                #will screen the misspelled words in the page (words that aren't found
                #within the more extensive "english_dict", here based on a 143K-word
                #dictionary "wlist_match7.txt") to see if any of them could be corrected by
                #altering one letter, and that the resulting word could be found in
                #the 15K-unambiguous word dictionary "short_english_dict", itself
                #derived from the 27K-word dictionary "wlist_match12.txt". If so,
                #the substitution would be made.
                #Also, the corrected word would only be in uppercase if every letter
                #of the misspelled word was in uppercase as well. Otherwise, the
                #corrected word will be in lowercase. The autocorrection steps
                #need to be performed before substitution of quotes and double quotes
                #to their RTF escape counterparts, as the "TextBlob" spellchecker
                #might recognize them and apply corrections, but this would likely
                #not be the case with the RTF escapes in their stead.
                if basic_autocorrect_lower == True:
                    word_list = re.split(r'(\W+)', text)
                    for j in range(len(word_list)):
                        if (word_list[j].isalpha() and word_list[j][0].islower() and
                        len(word_list[j]) > 4 and word_list[j] + "\n" not in english_dict):
                            len_misspelled_word = len(word_list[j])
                            start_misspelled_word = word_list[j][:3]
                            end_misspelled_word = word_list[j][-3:]
                            matching_words = [w.strip() for w in short_english_dict if
                            len(w.strip()) == len_misspelled_word and (w.strip()[:3] == start_misspelled_word or
                            w.strip()[-3:] == end_misspelled_word)]
                            for k in range(len(matching_words)):
                                matching_letters_counter = 0
                                for l in range(len(matching_words[k])):
                                    if matching_words[k][l] == word_list[j][l]:
                                        matching_letters_counter += 1
                                if matching_letters_counter == len_misspelled_word-1 and word_list[j].isupper():
                                    word_list[j] = matching_words[k].upper()
                                    break
                                elif matching_letters_counter == len_misspelled_word-1:
                                    word_list[j] = matching_words[k]
                                    break
                    corrected_text = "".join(word_list)


                #If the user has selected the "basic_autocorrect_lower" option, then the code
                #will screen the misspelled words in the page (words that aren't found
                #within the more extensive "english_dict", here based on a 143K-word
                #dictionary "wlist_match7.txt") to see if any of them could be corrected by
                #altering one letter, and that the resulting word could be found in
                #the 15K-unambiguous word dictionary "short_english_dict", itself
                #derived from the 27K-word dictionary "wlist_match12.txt". If so,
                #the substitution would be made.
                #Also, the corrected word would be in uppercase if every letter
                #of the misspelled word was in uppercase as well. If only the first
                #letter of the misspelt word was in uppercase, then the corrected
                #word would be capitalized. Otherwise, the corrected word will be
                #in lowercase.
                elif basic_autocorrect == True:
                    word_list = re.split(r'(\W+)', text)
                    for j in range(len(word_list)):
                        if (word_list[j].isalpha() and len(word_list[j]) > 4 and
                        word_list[j] + "\n" not in english_dict):
                            len_misspelled_word = len(word_list[j])
                            start_misspelled_word = word_list[j][:3]
                            end_misspelled_word = word_list[j][-3:]
                            matching_words = [w.strip() for w in short_english_dict if
                            len(w.strip()) == len_misspelled_word and (w.strip()[:3] == start_misspelled_word or
                            w.strip()[-3:] == end_misspelled_word)]
                            for k in range(len(matching_words)):
                                matching_letters_counter = 0
                                for l in range(len(matching_words[k])):
                                    if matching_words[k][l] == word_list[j][l]:
                                        matching_letters_counter += 1
                                if matching_letters_counter == len_misspelled_word-1 and word_list[j].isupper():
                                    word_list[j] = matching_words[k].upper()
                                    break
                                elif matching_letters_counter == len_misspelled_word-1 and word_list[j][0].isupper():
                                    word_list[j] = matching_words[k].capitalize()
                                    break
                                elif matching_letters_counter == len_misspelled_word-1:
                                    word_list[j] = matching_words[k]
                                    break
                    corrected_text = "".join(word_list)


                #If the user has selected the "autocorrect" option, then the code
                #will screen all of the words against the TextBlob "spellcheck()" method,
                #and substitutions will only be done if the confidence is above 90% that
                #the suggestion is the correct one. Also, the corrected word would be
                #in uppercase if every letter of the misspelled word was in uppercase
                #as well. If only the first letter of the misspelt word was in uppercase,
                #then the corrected word would be capitalized. Otherwise, the corrected
                #word will be in lowercase.
                elif autocorrect == True:
                    word_list = re.split(r'(\W+)', text)
                    for j in range(len(word_list)):
                        if word_list[j].isalpha() and word_list[j] != "qc":
                            word_suggestion = Word(word_list[j]).spellcheck()[0]
                            if word_suggestion[1] >= autocorrect_confidence:
                                if word_list[j].isupper():
                                    word_list[j] = word_suggestion[0].upper()
                                elif word_list[j][0].isupper():
                                    word_list[j] = word_suggestion[0].capitalize()
                                else:
                                    word_list[j] = word_suggestion[0]
                    corrected_text = "".join(word_list)


                #If there is at least one instance of a non-directional single "'" or
                #double '"' quote in the "text" string, or if the user has selected the
                #"smart_quotes" or "symmetrical_quotes" options, then all of the directional
                #quotes found within the page are switched to their non-directional counterparts.
                #This would be relevant if the user has trained the CNN model on directional
                #quotes, but didn't get good OCR accuracy when handling them. Alternatively,
                #if the user has trained their model on directional quotes but wants to have
                #symmetrical quotes in the final document, the directional quotes, if present,
                #still need to be changed for the symmetrical quotes. After that first step,
                #if there was at least one instance of a symmetrical quote in the document and
                #that the user didn't specify the "symmetrical_quotes" option, or if they
                #selected the "smart_quotes" option, the appropriate directional quotes will
                #be applied to the page.
                if (smart_quotes == True or symmetrical_quotes == True or
                (text.find("'") != -1 or text.find('"') != -1)):
                    text = (text.replace("‘", "'").replace("’", "'")
                    .replace('“', '"').replace('”', '"'))
                    #The same modifications with respect to smartquotes and RTF escapes need to be
                    #done to the corrected text string (if the user has selected this option), as
                    #the correction step needs to be performed before changing any character to its
                    #corresponding RTF escape, which could lead to exclusions when performing the
                    #autocorrect with the TextBlob module.
                    if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                        corrected_text = (corrected_text.replace("‘", "'").replace("’", "'")
                        .replace('“', '"').replace('”', '"'))

                    #The symmetrical quotes will be changed to their directional counterparts,
                    #if the user hasn't passed in the "symmetrical_quotes" argument, in which
                    #case the final document will have the symmetrical quotes.
                    if symmetrical_quotes == False:
                        #The nested quotes and single or double quotes followed by a space (and thus mapping
                        #to closing directional quotes) are changed to their RTF escape equivalents.
                        quote_substitutions = [['"' + "'", r"\'93" + r"\'91"], ["'" + '"', r"\'92" + r"\'94"],
                        ["' ", r"\'92" + ' '], ['" ', r"\'94" + ' ']]
                        for quote in quote_substitutions:
                            text = re.sub(quote[0], quote[1], text)
                            if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                                corrected_text = re.sub(quote[0], quote[1], corrected_text)

                        #If the first character of the "text" string
                        #is a quote, it is then changed for the corresponding
                        #opening directional quote.
                        if text[0] == "'":
                            text = r"\'91" + text[1:]
                        elif text[0] == '"':
                            text = r"\'93" + text[1:]
                        if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                            if corrected_text[0] == "'":
                                corrected_text = r"\'91" + corrected_text[1:]
                            elif corrected_text[0] == '"':
                                corrected_text = r"\'93" + corrected_text[1:]


                        #If the last character of the "text" string
                        #is a quote, it is then changed for the corresponding
                        #closing directional quote.
                        if text[-1] == "'":
                            text = text[:-1] + r"\'92"
                        elif text[-1] == '"':
                            text = text[:-1] + r"\'94"
                        if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                            if corrected_text[-1] == "'":
                                corrected_text = corrected_text[:-1] + r"\'92"
                            elif corrected_text[-1] == '"':
                                corrected_text = corrected_text[:-1] + r"\'94"

                        #The indices of any remaining symmetrical double quotes ('"') are stored in
                        #the list "double_quote_indices" and cycled through using a "for" loop.
                        #If the index is above zero and smaller than the last index of the "double_quote_indices"
                        #list, and if the preceding character is not a space, "(", "[", "{", "-", "_" then the
                        #closing directional quote is substituted for the symmetrical one. The "text"
                        #string is updated by slicing it while skipping over what was the symmetrical double
                        #quote ('"') at index "double_quote_indices[i]" in "text". The "for" loop proceeds
                        #in reverse order to avoid indexing issues when substituting quotes for multi-character RTF escapes.
                        double_quote_matches = re.finditer('"', text)
                        double_quote_indices = [match.start() for match in double_quote_matches]
                        for i in range(len(double_quote_indices)-1, -1, -1):
                            if (double_quote_indices[i] > 0 and double_quote_indices[i] < len(text)-1 and
                            text[double_quote_indices[i]-1] not in [" ", "(", "[", "{", "-", "_"]):
                                text = (text[:double_quote_indices[i]] + r"\'94" +
                                text[double_quote_indices[i]+1:])


                            #If the index is above zero and smaller than the last index of the "double_quote_indices"
                            #list, and if the previous character is not a letter and the following character is either
                            #a letter or "¡", "¿" (which would start an exclamation or question, respectively, in Spanish)
                            #a backslash (if the quote is followed by an RTF command or escape such as "\i"),
                            #or a smallcaps (the italics will be dealt with after this step), then the double quote
                            #is changed to the opening directional quote.
                            elif (double_quote_indices[i] > 0 and double_quote_indices[i] < len(text)-1 and
                            (text[double_quote_indices[i]-1].isalpha() == False and
                            (text[double_quote_indices[i]+1].isalpha() or
                            text[double_quote_indices[i]+1] in ["¡", "¿", "\\", "_"]))):
                                text = (text[:double_quote_indices[i]] + r"\'93" +
                                text[double_quote_indices[i]+1:])


                        if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                            double_quote_matches = re.finditer('"', corrected_text)
                            double_quote_indices = [match.start() for match in double_quote_matches]
                            for i in range(len(double_quote_indices)-1, -1, -1):
                                if (double_quote_indices[i] > 0 and double_quote_indices[i] < len(corrected_text)-1 and
                                corrected_text[double_quote_indices[i]-1] not in [" ", "(", "[", "{", "-", "_"]):
                                    corrected_text = (corrected_text[:double_quote_indices[i]] + r"\'94" +
                                    corrected_text[double_quote_indices[i]+1:])
                                elif (double_quote_indices[i] > 0 and double_quote_indices[i] < len(corrected_text)-1 and
                                (corrected_text[double_quote_indices[i]-1].isalpha() == False and
                                (corrected_text[double_quote_indices[i]+1].isalpha() or
                                corrected_text[double_quote_indices[i]+1] in ["¡", "¿", "\\", "_"]))):
                                    corrected_text = (corrected_text[:double_quote_indices[i]] + r"\'93" +
                                    corrected_text[double_quote_indices[i]+1:])


                        single_quote_matches = re.finditer("'", text)
                        single_quote_indices = [match.start() for match in single_quote_matches]
                        for i in range(len(single_quote_indices)-1, -1, -1):
                            #The "if" statement will also change the symmetrical single quote to the closing
                            #directional single quote in contractions such as "don't", as only the preceding character
                            #is considered. In this case, the preceding character must not be a space, "(", "[", "{",
                            #"-", "_" nor a backslash (so that the single quote in the RTF escapes (such as r"\'92"))
                            #are not confused for actual single quotes.
                            if (single_quote_indices[i] > 0  and single_quote_indices[i] < len(text)-1 and
                            text[single_quote_indices[i]-1] != "\\" and text[single_quote_indices[i]-1] not in
                            [" ", "(", "[", "{", "-", "_",  "\\"]):
                                text = (text[:single_quote_indices[i]] + r"\'92" +
                                text[single_quote_indices[i]+1:])
                            elif (single_quote_indices[i] > 0 and single_quote_indices[i] < len(text)-1 and
                            (text[single_quote_indices[i]-1] != "\\" and
                            text[single_quote_indices[i]-1].isalpha() == False and
                            (text[single_quote_indices[i]+1].isalpha() or
                            text[single_quote_indices[i]+1] in ["¡", "¿", "\\", "_"]))):
                                text = (text[:single_quote_indices[i]] + r"\'91" +
                                text[single_quote_indices[i]+1:])

                        if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                            single_quote_matches = re.finditer("'", corrected_text)
                            single_quote_indices = [match.start() for match in single_quote_matches]
                            for i in range(len(single_quote_indices)-1, -1, -1):
                                if (single_quote_indices[i] > 0  and single_quote_indices[i] < len(corrected_text)-1 and
                                corrected_text[single_quote_indices[i]-1] != "\\" and corrected_text[single_quote_indices[i]-1] not in
                                [" ", "(", "[", "{", "-", "_",  "\\"]):
                                    corrected_text = (corrected_text[:single_quote_indices[i]] + r"\'92" +
                                    corrected_text[single_quote_indices[i]+1:])
                                elif (single_quote_indices[i] > 0 and single_quote_indices[i] < len(corrected_text)-1 and
                                (corrected_text[single_quote_indices[i]-1] != "\\" and
                                corrected_text[single_quote_indices[i]-1].isalpha() == False and
                                (corrected_text[single_quote_indices[i]+1].isalpha() or
                                corrected_text[single_quote_indices[i]+1] in ["¡", "¿", "\\", "_"]))):
                                    corrected_text = (corrected_text[:single_quote_indices[i]] + r"\'91" +
                                    corrected_text[single_quote_indices[i]+1:])

                #The RTF escapes are substituted for the symbols to allow for adequate representation within
                #the RTF file. It is important to do the autocorrect step before the substitution of RTF escapes
                #for their original characters, as some RTF escapes end with a letter, which could be included
                #in an adjoining word when splitting the words of the text string in the autocorrect code:
                #(word_list = re.split(r'(\W+)', text))
                rtf_escapes = [['…', r"\'85"], ['†', r"\'86"], ['‡', r"\'87"],  ['✕', r"\'d7"], ['\+', r"\'2b"],
                ['⋅', r"\'b7"], ['·', r"\'b7"], ['÷', r"\'f7"], ['/', r"\'2f"], ['>', r"\'3e"], ['<', r"\'3c"],
                ['¢', r"\'a2"], ['\$', r"\'24"], ['€', r"\'80"], ['¤', r"\'a4"], ['¥', r"\'a5"],  [r'\[', r"\'5b"],
                ['\]', r"\'5d"], ['\^', r"\'5e"], ['ˆ', r"\'88"], ['`', r"\'60"], ['´', r"\'b4"], ['”', r"\'94"],
                ['\|', r"\'7c"], ['¦', r"\'a6"], ['£', r"\'a3"], ['″', r"\'22"], ['%', r"\'25"], ['‰', r"\'89"],
                ['Š', r"\'8a"], ['š', r"\'9a"], ['‹', r"\'8b"], ['›', r"\'9b"], ['Œ', r"\'8c"], ['œ', r"\'9c"],
                ['Ÿ', r"\'9f"], ['Ž', r"\'8e"], ['ž', r"\'9e"], ['°', r"\'b0"], ['#', r"\'23"], ['&', r"\'26"],
                ['©', r"\'a9"], ['™', r"\'99"], ['•', r"\'95"], ['@', r"\'40"], [r'\*', r"\'2a"], ['∼', r"\'98"],
                ['±', r"\'b1"], ['€', r"\'80"], ['͵', r"\'80"], ['ƒ', r"\'83"], ['„', r"\'84"], ['〃', r"\'22"],
                ['¶', r"\'b6"], ['®', r"\'ae"], ['§', r"\'a7"], ['«', r"\'ab"], ['»', r"\'bb"], ['¡', r"\'a1"],
                ['¿', r"\'bf"], ['¨', r"\'a8"], ['ª', r"\'aa"], ['º', r"\'ba"], ['¬', r"\'ac"], ['¯', r"\'af"],
                ['¼', r"\'bc"], ['½', r"\'bd"], ['¾', r"\'be"], ['¹', r"\'b9"], ['²', r"\'b2"], ['³', r"\'b3"],
                ['µ', r"\'b5"], ['¸', r"\'b8"], ['À', r"\'c0"], ['Á', r"\'c1"], ['Â', r"\'c2"], ['Ã', r"\'c3"],
                ['Ä', r"\'c4"], ['Å', r"\'c5"], ['Æ', r"\'c6"], ['Ç', r"\'c7"], ['È', r"\'c8"], ['É', r"\'c9"],
                ['Ê', r"\'ca"], ['Ë', r"\'cb"], ['Ì', r"\'cc"], ['Í', r"\'cd"], ['Î', r"\'ce"], ['Ï', r"\'cf"],
                ['Ð', r"\'d0"], ['Ñ', r"\'d1"], ['Ò', r"\'d2"], ['Ó', r"\'d3"], ['Ô', r"\'d4"], ['Õ', r"\'d5"],
                ['Ö', r"\'d6"], ['Ø', r"\'d8"], ['Ù', r"\'d9"], ['Ú', r"\'da"], ['Û', r"\'db"], ['Ü', r"\'dc"],
                ['Ý', r"\'dd"], ['Þ', r"\'de"], ['ß', r"\'df"], ['à', r"\'e0"], ['á', r"\'e1"], ['â', r"\'e2"],
                ['ã', r"\'e3"], ['ä', r"\'e4"], ['å', r"\'e5"], ['æ', r"\'e6"], ['ç', r"\'e7"], ['è', r"\'e8"],
                ['é', r"\'e9"], ['ê', r"\'ea"], ['ë', r"\'eb"], ['ì', r"\'ec"], ['í', r"\'ed"], ['î', r"\'ee"],
                ['ï', r"\'ef"], ['ð', r"\'f0"], ['ñ', r"\'f1"], ['ò', r"\'f2"], ['ó', r"\'f3"], ['ô', r"\'f4"],
                ['õ', r"\'f5"], ['ö', r"\'f6"], ['ø', r"\'f8"], ['ù', r"\'f9"], ['ú', r"\'fa"], ['û', r"\'fb"],
                ['ü', r"\'fc"], ['ý', r"\'fd"], ['þ', r"\'fe"], ['ÿ', r"\'ff"], ["\-", r"\'2d"], ["—", r"\'97"],
                ['—', r"\'96"], ['_', r"\'5f"], ["‘", r"\'91"], ["’", r"\'92"], ['“', r"\'93"], ['=', r"\'3d"],
                ['–', r"\'96"]]
                for escape in rtf_escapes:
                    text = re.sub(escape[0], escape[1], text)
                    if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                        corrected_text = re.sub(escape[0], escape[1], corrected_text)

                #Successive spaces in excess of two (to accomodate for any optional spaces after RTF commands)
                #are changed for a single space, and the extraneous spaces before closing double or single quotes
                #are removed. Also, two successive hyphens are changed for an "em" dash.
                text = (re.sub('[" "]{2,}', " ", text).replace(" \\'94", "\\'94")
                .replace(" \\'92", "\\'92").replace("\\'2d\\'2d", r"\'97"))
                if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                    corrected_text = (re.sub('[" "]+', " ", corrected_text).replace(" \\'94", "\\'94")
                    .replace(" \\'92", "\\'92").replace("--", r"\'97"))
                    with open(os.path.join(path, OCR_text_file_name + '-OCR (autocorrect).rtf'), 'a+', encoding="utf-8") as e:
                        e.write(corrected_text)
                f.write(text)

            #An ".rtf" file was created and a basic document prolog was added earlier, followed by the
            #contents of the "text" string. The closing curly bracket (}) is now added at the very end
            #of the ".rtf" document, as the "for JPEG_file_name in JPEG_file_names" loop is complete. The "}"
            #matches the first "{" of the prolog.
            if basic_autocorrect == True or basic_autocorrect_lower == True or autocorrect == True:
                with open(os.path.join(path, OCR_text_file_name + '-OCR (autocorrect).rtf'), 'a+', encoding="utf-8") as e:
                    e.write("}")
            f.write("}")
