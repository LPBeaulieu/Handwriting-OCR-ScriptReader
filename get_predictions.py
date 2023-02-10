import cv2
import os
import re
from alive_progress import alive_bar
import numpy as np
from fastai.vision.all import *


#This function extracts the x and y coordinates for every character in a typewritten text image, allong
#with some of the space character (" ") indices (see explanations below).
def get_character_x_y_coordinates(image):
    imgheight=image.shape[0]
    imgwidth=image.shape[1]
    chars_x_y_coordinates = []
    lines_y_min_maxes = [[]]
    # The image's numpy array is filtered using the np.where() function to convert white pixels (255) to 0
    # and non-white pixels (!= 255) to 1. The rows are added up (summation along the 1 axis) to determine
    # how many non-white pixels there are for a given y coordinate. The y coordinates where the number
    # of non-white pixels exceed 200 are extracted using the np.where() function. The indice where two lines
    # should be split are determined using another np.where() function, if there are over 50 pixels on the y axis
    # separating two elements of the "y_pixels" array. The "y_pixels" array is then split along those indices+1
    # using the np.split() function and the "lines_y_min_maxes" array is populated, and then overwritten with
    # the y coordinate minimum and maximum for every line. Then the vertical center of every line is determined,
    # and the vertical margins for every line are normalized based on the "character_width".
    # A similar procedure is used for the screening along the x axis, without the normalization step, as there are
    # further processing steps required before character width normalization.
    image_filtered = np.where(image == 255, 0, 1)
    y_pixels = np.sum(image_filtered, axis=1)
    y_pixels = np.where(y_pixels > 200)[0]
    y_split_indices = np.where([y_pixels[i]-y_pixels[i-1] > 50 for i in range(1, len(y_pixels))])[0]
    lines_y_min_maxes = np.split(y_pixels, y_split_indices+1)
    lines_y_min_maxes = [[lines_y_min_maxes[i][0], lines_y_min_maxes[i][-1]] for i in range(len(lines_y_min_maxes))]
    chars_y_min_maxes = [[int(((lines_y_min_maxes[i][-1] - lines_y_min_maxes[i][0])/2 + lines_y_min_maxes[i][0])
                        - rectangle_half_height), int(((lines_y_min_maxes[i][-1] - lines_y_min_maxes[i][0])/2
                        + lines_y_min_maxes[i][0]) + rectangle_half_height)] for i in range(len(lines_y_min_maxes))]

    for char_y_min_max in chars_y_min_maxes:
        #The if statement excludes detected characters if they are overlapping with the upper or lower borders
        #of the page. The scanned image might have darkened edges, as the scanner plate is larger than the page.
        #Also, the if statement excludes overlapping lines (of which the difference between the y minima is less
        #than 2 times "character_width"). 50 pixels are being subtracted from y_min and 50 are added to y_max
        #to make up for the fact that the rectangle height will be increased later on in the code. Of note,
        #the segmentation algorithm works just as well with a flatbed scanner or a multi-sheet scanner.
        line_index = chars_y_min_maxes.index(char_y_min_max)
        if ((char_y_min_max[0] - 50 <= 0 or char_y_min_max[1] + 50 >= imgheight) or
        (line_index > 0 and char_y_min_max[0]-chars_y_min_maxes[line_index-1][0] < 2*character_width)):
            pass
        else:
            line_image = np.sum(image_filtered[char_y_min_max[0]:char_y_min_max[1], 0:imgwidth], axis=0)
            #As fresh typewriter ink ribbons lead to darker text and more ink speckling on the page, in the
            #presence of dark typewritten text you should decrease the segmentation sensitivity
            #(increase the number of non-white y pixels required for a given x coordinate in order for
            #that x coordinate to be included in the segmentation). That is to say that on a fresh ribbon
            #of ink, you should increase the value of 3 to about 6 (results will vary based on your
            #typewriter's signal to noise ratio) in the line 57 in order to avoid including unwanted noise
            #in the character rectangles.
            x_pixels = np.where(line_image >= 3)[0]
            x_split_indices = np.where([x_pixels[i]-x_pixels[i-1] > 10 for i in range(1, len(x_pixels))])[0]
            chars_x_min_maxes = np.split(x_pixels, x_split_indices+1)
            #The "chars_x_min_maxes" list is overwritten with a list of the minimum and maximum x coordinates
            #for a given character. The if statement excludes detected characters for which there are less than
            #10 x coordinates meeting the conditions above, and also excludes detected characters if their
            #x minimum -50 is below 0 (before the beginning of the page) or their x maximum +50 is above
            #"imgwidth" (the detected character spills over the right edge of the page), to avoid any artifacts
            #originating from document scanning. 50 pixels are being subtracted from x_min and 50 are added
            #to x_max to make up for the fact that the rectangle width will be increased later on in the code.
            chars_x_min_maxes = [[chars_x_min_maxes[i][0], chars_x_min_maxes[i][-1]]
                                    for i in range(len(chars_x_min_maxes)) if len(chars_x_min_maxes[i])>10 and
                                    (chars_x_min_maxes[i][0] - 50 > 0 and chars_x_min_maxes[i][1] + 50 < imgwidth)]
            new_chars_x_min_maxes = [[]]
            rectangle_index = 0
            for char_x_min_max in chars_x_min_maxes:
                #If a rectangle is wider than 1.5 times the "character_width", it will be split up into
                #individual character rectangles. The updated x coordinates [minimum, maxmum] are stored
                #in the list of lists "new_chars_x_min_maxes", in order to substitute the coordinates
                #of the several individually split characters for the coordinates of the wide
                #multi-character rectangle in "chars_x_min_maxes".
                rectangle_width = char_x_min_max[1]-char_x_min_max[0]
                if rectangle_width > 1.5*character_width:
                    number_of_characters = round(rectangle_width/character_width)
                    for n in range(number_of_characters):
                        if n == 0:
                            new_chars_x_min_maxes = ([[char_x_min_max[0], char_x_min_max[0] +
                                character_width]])
                            new_x_min = char_x_min_max[0] + character_width + spacer_between_characters
                        else:
                            new_x_max = new_x_min + character_width
                            new_chars_x_min_maxes.append([new_x_min, new_x_max])
                            new_x_min = new_x_max + spacer_between_characters
                    #The wide multi-character rectangle coordinates are replaced by those of the
                    #individually split characters in "chars_x_min_maxes" and the "rectangle_index"
                    #is incremented to take into account that one big rectangle has been replaced
                    #by several smaller rectangles.
                    chars_x_min_maxes = (chars_x_min_maxes[:rectangle_index] +
                    new_chars_x_min_maxes + chars_x_min_maxes[rectangle_index+1:])
                    rectangle_index += number_of_characters
                else:
                    rectangle_index += 1

            #The following code is to generate rectangles of uniform pixel count (width and height) for the
            #characters, based on the calculated center of the characters. Note that the iteration through
            #the x min and max coordinates of every character starts in reverse order (from the last
            #character of every line) to avoid indexing issues when removing overlapping characters
            #from the list ("chars_x_min_maxes.pop(character_counter)").
            character_counter = len(chars_x_min_maxes)-1
            for char_x_min_max in sorted(chars_x_min_maxes, reverse=True):
                character_center_x = int((char_x_min_max[1]-char_x_min_max[0])/2) + char_x_min_max[0]
                chars_x_min_maxes[character_counter] = [character_center_x - rectangle_half_width,
                    character_center_x + rectangle_half_width]
                #This if statement removes any overlaping characters, of which the difference between
                #their x_minima is lower or equal to 0.40 times "character_width". You might need to
                #alter the values of the variables "character_width" (default value of 55 pixels for
                #8 1/2" x 11" typewritten pages scanned at a resolution of 600 dpi) and
                #"spacer_between_character" default value of 5 pixels, as your typewriter may have a
                #different typeset than those of my typewriters (those two defult parameters were suitable for
                #both my 2021 Royal Epoch and 1968 Olivetti Underwood Lettra 33 typewriters). These
                #parameters ("character_width", "spacer_between_characters" and "line_image >= 3"
                #(see comment above) should be adjusted in the same way in all the Python code files
                #(except "train_model.py", where they are absent) to ensure consistent segmentation
                #in all steps of the process.
                if (character_counter > 0 and (chars_x_min_maxes[character_counter][0] -
                chars_x_min_maxes[character_counter-1][0]) <= 0.40*character_width):
                    chars_x_min_maxes.pop(character_counter)
                character_counter-=1


            #Drawing the rectangles in green on a copy of the image in the "Page image files with rectangles"
            #folder to check whether the coordinates of the characters line up well. Also, the list
            #"chars_x_y_coordinates" is populated with the character coordinates. Extra pixels on the
            #'x' axis are added on either side of the character to make sure that in the vast majority of cases,
            #the character will be fully included in the frame. This adjustment needs to be done after
            #removing overlaping characters (which is why I didn't write
            #"character_center_x - rectangle_half_width -20, character_center_x + rectangle_half_width + 20" above)
            for char_x_min_max in chars_x_min_maxes:
                (chars_x_y_coordinates.append([[char_x_min_max[0]-20, char_y_min_max[0]],
                    [char_x_min_max[1]+20, char_y_min_max[1]]]))
                (cv2.rectangle(text_image_copy, (char_x_min_max[0], char_y_min_max[0]),
                    (char_x_min_max[1], char_y_min_max[1]), (0,255,0),3))

    if not os.path.exists(cwd + "/Page image files with rectangles/"):
        os.makedirs(cwd + "/Page image files with rectangles/")
    (cv2.imwrite(cwd + '/Page image files with rectangles/' + image_name[:-4] +
    ' with character rectangles.jpg', text_image_copy))

    return chars_x_y_coordinates



os.system('clear')
cwd = os.getcwd()
#Get a list of ".jpg" file names in folder "OCR Raw Data"
image_names = [file_name for file_name in sorted(os.listdir(cwd + "/OCR Raw Data/")) if file_name[-4:] == ".jpg"]

#Generate cropped character images from the image files listed in the "image_names" list and store
#them in an image folder. These cropped character images will be deleted further on in the code (see comments below)
#and the image folder name is extracted from the first image name in the "image_names" list, including all
#characters up to the last hyphen (e.g. "Alice's Adventures in Wonderland Chapter 1-0001.jpg" would
#give the following extracted name: "Alice's Adventures in Wonderland Chapter 1")
hyphen_matches = re.finditer("-", image_names[0])
hyphen_indices = []
for match in hyphen_matches:
    hyphen_indices.append(match.start())
OCR_text_file_name = image_names[0][:hyphen_indices[-1]]

path = cwd + "/OCR Predictions/" + OCR_text_file_name + "/"
if not os.path.exists(path):
    os.makedirs(path)

#Import the convoluted neural network (cnn) deep learning model for OCR prediction.
#My optimal model trained on 26 pages of typewritten text using a 1968 Olivetti Underwood Lettra 33 typewriter,
#with a batch size of 64, a learning rate of 0.005 and 3 epochs of training yielded a validation accuracy
#of 99.96%. The average validation accuracy for 4 training runs with the same hyperparameters mentioned above
#is 99.90% and the optimal model (99.96%) was selected and named Model_Olivetti_1968_Underwood_Lettra_33_acc9996
learn = load_learner(cwd + '/typewriter_OCR_cnn_model')

with alive_bar(len(image_names)) as bar:
    with open(path + OCR_text_file_name + '-OCR.rtf', 'a+') as f:
        #An ".rtf" file is created and a basic document prolog is added. The closing curly bracket (}) is
        #added at the very end of the code, when the "for image_name in image names" loop is complete.
        #For further information on ".rtf" commands, please consult the RTF Pocket Guide, available
        #free of charge online at the following address:
        #https://www.oreilly.com/library/view/rtf-pocket-guide/9781449302047/ch01.html
        f.write(r"{\rtf1 \ansi \deff0 {\fonttbl {\f0 Ubuntu;}} \f0 \fs24 \par ")
        for image_name in image_names:
            #Insert a space at the beginning of every page after the first "image_name".
            if image_names.index(image_name)>0:
                f.write(" ")

            print(f'\nCurrently processing image entitled: "{image_name}"\n')
            text_image = cv2.imread('OCR Raw Data/' + str(image_name))
            text_image_copy = text_image.copy()
            #Convert image from RGB to grayscale
            text_image_gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)
            #Apply binary thresholding (pixels with a value above the first number will be set to 255 (white))
            #Having a relatively high threshold at 245 ensures that even faint text is detected and a
            #threshold of 240 is selected for cropping to avoid having fuzzy characters for OCR.
            ret, threshold = cv2.threshold(text_image_gray, 240, 255, cv2.THRESH_BINARY)
            text_image_threshold_240 = threshold.copy()
            ret, threshold = cv2.threshold(text_image_gray, 245, 255, cv2.THRESH_BINARY)
            text_image_threshold_245 = threshold.copy()


            '''CHARACTER WIDTH AND SPACING PARAMETERS'''
            #You might need to adjust the "character width" and "spacer_between_characters"
            #according to your typewriter. These two figures seem to work well with a
            #1968 Olivetti Underwood Lettra 33 typewriter and a scanned image of a
            #letter page (8 1/2" x 11") at 600 dpi resolution. A "spacer_between_characters"
            #is needed to avoid the rectangles being progressively staggered relative to the characters,
            #when splitting a multi-character wide rectangle (see code in function above).
            character_width = 55
            spacer_between_characters = 5

            rectangle_half_width = int(character_width*3/5)
            rectangle_half_height = int(character_width*3/2)

            #Extracting the data returned from the "get_character_coordinates" function.
            chars_x_y_coordinates = get_character_x_y_coordinates(text_image_threshold_245)

            #Gathering the indices of spaces within the "space_indices" list, if the difference between the
            #x-axis minimum of a character (+20 pixels to give what was originally the character rectangle)
            #and the x-axis maximum (-20 pixels for the same reason) of the preceding character exceeds 45%
            #of the "character_width". Of note, not every space index is included here, as some of the
            #actual spaces in the scanned text are detected as characters by the code. However, the
            #machine learning OCR code should assign these "blank" characters as spaces. The sum
            #of the space information from the character detection (segmentation) and the machine
            #learning processes should locate most spaces. If any spaces are unaccounted for,
            #it would likely result in unknown compound words, which could readily be identified
            #by red squiggly lines in the word processor.
            space_indices = []
            space_index_correction_factor = 0
            for i in range(1, len(chars_x_y_coordinates)):
                pixels_between_chars = (chars_x_y_coordinates[i][0][0]+20) - (chars_x_y_coordinates[i-1][1][0]-20)
                if (chars_x_y_coordinates[i][0][1] != chars_x_y_coordinates[i-1][0][1] or
                pixels_between_chars > 0.45*character_width):
                    space_indices.append(i + space_index_correction_factor)
                    space_index_correction_factor+=1

            char_files = []
            for i in range(len(chars_x_y_coordinates)):
                (cv2.imwrite(path + str(i) + ".jpg",
                    text_image_threshold_240[chars_x_y_coordinates[i][0][1]:chars_x_y_coordinates[i][1][1],
                    chars_x_y_coordinates[i][0][0]:chars_x_y_coordinates[i][1][0]]))
                char_files.append(path + str(i) + ".jpg")

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
            text = [learn.dls.vocab[preds_argmax[i]] for i in range(len(preds_argmax))]

            #If you want to print out the dictionary mapping the labels to the label
            #indices, uncomment the following line:
            # print(learn.dls.vocab.o2i)

            #Once the "text" list of predicted characters has been populated, delete the individual
            #character ".jpg" images used for OCR (you can comment out the following lines of
            #code should you want to retain them for troubleshooting purposes).
            for i in range(len(text)):
                os.remove(path + str(i) + '.jpg')

            #Substitute the actual character labels for the labels that were written in long
            #form for compatibility reasons.
            quote_open = False
            for i in range(len(text)-1, -1, -1):
                #If the label is "empty" (a typo overlaid with a hashtag symbol),
                #replace those characters with " ". Superfluous spaces are removed at the end of the code
                #(text = "".join(text).replace("  ", " ")).
                if text[i] == "empty":
                    text[i] = ""
                #If the label is a "space", it is replaced with " ".
                elif text[i] == "space":
                    text[i] = " "
                #If the label is "forward slash", it is replaced with a forward slash.
                elif text[i] == "forward slash":
                    text[i] = "/"
                #If the label is "period", it is replaced with ".".
                elif text[i] == "period":
                    text[i] = "."

            #Add in the spaces from the "space_indices" list
            for i in range(len(space_indices)):
                text.insert(space_indices[i], " ")

            for i in range(0, len(text)):
                #Replace "=" with a backslash if the following list element isn't "double quote",
                #and if that element is either a letter, an empty string (in case there were
                #hashtag typos in between the "=" and the rtf command or escape, e.g. "=##i")
                #or one of the rtf escape symbols:
                #(\' (ASCII rtf character escape), \- (hyphenation point) or \_ (nonbreaking hyphen))
                if (i < len(text)-1 and text[i] == "=" and text[i+1]!= "double quote" and
                (text[i+1].isalpha() or text[i+1] in ["", "'", "-", "_"])):
                    text[i] = "\\"
                #If the label is "double quote" or "'", it is replaced  with the corresponding
                #directional quote mark. This needs to be done after all the spaces are in place
                #in the list "text", as the directionality of the quote marks depend on the
                #presence of spaces. If the "double quote" character is the last character
                #in the page or if there is a space after "double quote", it will be replaced
                #with a closing double quote. Otherwise, it will be replaced with an opening double quote.
                #Conversely, if the single quote is the first character in the page, or if
                #it is preceded by a space, it will be replaced with an opening single quote.
                #Otherwise, it will be replaced with a closing single quote.
                elif text[i] == "double quote":
                    if i == len(text)-1 or text[i+1] == " ":
                        text[i] = "\\'94"
                    else:
                        text[i] = "\\'93"
                elif text[i] == "'" and text[i-1] != "\\":
                    if i == 0 or text[i-1] == " ":
                        text[i] = "\\'91"
                    else:
                        text[i] = "\\'92"


                #Replace lowercase "l" or uppercase "O" by the digits one and zero, respectively if
                #there is a digit either directly before or after the character, or if the character
                #before or after is either ":", "," or "." and the character before and after the
                #separator is another digit (such as 2.0 or 0.2).
                elif (text[i] == "l" and ((i>0 and text[i-1].isdigit()) or (i<len(text)-1 and text[i+1].isdigit()) or
                (i<len(text)-2 and text[i+2].isdigit() and text[i+1] in [",", ".", ":"]) or
                (i>1 and text[i-2].isdigit() and text[i-1] in [",", ".", ":"]))):
                    text[i] = str(1)
                elif (text[i] == "O" and ((i>0 and text[i-1].isdigit()) or (i<len(text)-1 and text[i+1].isdigit()) or
                (i<len(text)-2 and text[i+2].isdigit() and text[i+1] in [",", ".", ":"]) or
                (i>1 and text[i-2].isdigit() and text[i-1] in [",", ".", ":"]))):
                    text[i] = str(0)

            '''DEFAULT (BASIC) RTF FORMATTING MODE'''
            #Join the elements of the "text" list and perform a series of string substitutions
            #to yield the final text and write it as a ".rtf" file. The following line replaces
            #every instance of at least one successive space with a single space, and then replaces
            #"\par" with "\par\pard", to set the new paragraph's formatting to the default settings.
            #This way, if you centered the previous paragraph (\qc), you don't need to remember
            #to set the next paragraph to the default alignment (\ql) by adding a "\pard".
            #For full rtf functionalities (without this last simplification), comment the next
            #line out and activate the line after it instead.
            text = (re.sub('[" "]+', " ", "".join(text)).replace(r"\par", r"\par\pard")
            .replace(r"\bO", r"\b0 ").replace(r"\iO", r"\i0 ").replace(r"\scapsO", r"\scaps0 ")
            .replace(r"\strikeO", r"\strike0 ").replace(r"\ulO", r"\ul0 ").replace(" .", ".")
            .replace(" ,", ",").replace(" :", ":").replace(" ;", ";").replace(" \\'94", "\\'94")
            .replace(" \\'92", "\\'92").replace("( ", "(").replace(" )", ")").replace(" ?", "?").replace(" !", "!"))

            '''ADVANCED RTF FORMATTING MODE'''
            #The following line of code is identical to the preceding one, except that there are no
            #substitutions of "\par\pard" for "\par", and curly brackets ({}) can be used in
            #formatting of the rtf document. Double parentheses will be converted to the
            #corresponding curly brackets. Should you want to use the advanced rtf formatting mode,
            #simply comment out the preceding line of code and activate the following one.
            # text = (re.sub('[" "]+', " ", "".join(text)).replace(r"\bO", r"\b0 ").replace(r"\iO", r"\i0 ")
            # .replace(r"\scapsO", r"\scaps0 ").replace(r"\strikeO", r"\strike0 ").replace(r"\ulO", r"\ul0 ")
            # .replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" ;", ";").replace(" \\'94", "\\'94")
            # .replace(" \\'92", "\\'92").replace(" ?", "?").replace(" !", "!").replace("( ", "(").replace(" )", ")")
            # .replace("((", "{").replace("))", "}"))


            f.write(text)
            bar()
        #An ".rtf" file was created and a basic document prolog was added earlier, followed by the
        #contents of the "text" string. The closing curly bracket (}) is now added at the very end
        #of the ".rtf" document, as the "for image_name in image names" loop is complete. The "}"
        #matches the first "{" of the prolog.
        f.write("}")
