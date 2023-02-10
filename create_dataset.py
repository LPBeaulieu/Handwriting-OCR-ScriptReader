import cv2
import os
import shutil
import re
from alive_progress import alive_bar
import numpy as np


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


#Clear the command line screen
os.system('clear')
#Get a list of ".jpg" file names in folder "Training&Validation Data"
cwd = os.getcwd()
image_names = [file_name for file_name in sorted(os.listdir(cwd + "/Training&Validation Data/")) if file_name[-4:] == ".jpg"]

#Generate cropped character images from the image files listed in the "image_names" list and store
#them in the "Dataset" folder.
with alive_bar(len(image_names)) as bar:
    char_index = 0
    for image_name in image_names:
        print(f'\nCurrently processing image entitled: "{image_name}"\n')
        text_image = cv2.imread('Training&Validation Data/' + str(image_name))
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

        #Building a list of character labels (labels were obtained from a txt file named after the "image_name",
        #which itself was written based on the image with overlaping character rectangles generated above).
        #An important note to those who would like to train their own models is to generate and save the txt
        #files exclusively in WordPad or Text Editor and not a full-fledged word processor, which would insert
        #formatting information that would skew the character count.
        labels = []
        with open('Training&Validation Data/' + image_name[:-4] + '.txt', 'r') as f:
            text = f.read()
            character_count = 0
            for char in text:
                #If the character in the txt file is a space or a line carriage ("\n"),
                #no label is appended to the list "labels".
                if text[character_count] == ' ' or text[character_count:character_count+1] == "\n":
                    character_count+=1
                    pass
                #If the character in the txt file is a "<" (Python source code u"\u003C"), "space"
                #is appended to the list "labels".
                elif text[character_count] == u"\u003C":
                    labels.append("space")
                    character_count+=1
                #If the character in the txt file is a "~" (Python source code u"\u007E"), "empty"
                #is appended to the list "labels".
                elif text[character_count] == u"\u007E":
                    labels.append("empty")
                    character_count+=1
                #If the character in the txt file is a "/" (Python source code u"\u002F"), "forward-slash"
                #is appended to the list "labels".
                elif text[character_count] == u"\u002F":
                    labels.append("forward slash")
                    character_count+=1
                #If the character in the txt file is a "?" (Python source code u"\u003F"), "?"
                #is appended to the list "labels".
                elif text[character_count] == u"\u003F":
                    labels.append(u"\u003F")
                    character_count+=1
                #If the character in the txt file is a ".", "period"
                #is appended to the list "labels".
                elif text[character_count] == '.':
                    labels.append("period")
                    character_count+=1
                #If the character in the txt file is a '"' (Python source code u"\u0022"), "double quote"
                #is appended to the list "labels".
                elif text[character_count] == u"\u0022":
                    labels.append("double quote")
                    character_count+=1
                #If the character in the txt file is a "@" (Python source code u"\u0040"), "to be deleted"
                #is appended to the list "labels".
                elif text[character_count] == u"\u0040":
                    labels.append("to be deleted")
                    character_count+=1
                #Other characters get appended to the list.
                else:
                    labels.append(text[character_count])
                    character_count+=1

        #As an added quality control step, the length of the list of character coordinates and their labels
        #are printed on screen. The two lengths should be equivalent, as they both refer to the same characters.
        print("Length of list 'chars_x_y_coordinates': " + str(len(chars_x_y_coordinates)))
        print("Length of list 'labels': " + str(len(labels)))

        #Generating the individual character images based on their x and y coordinates (from chars_x_y_coordinates),
        #every image being placed in a folder corresponding to its label (labels were obtained from a txt file
        #named after the "image_name"). The counter "char_index" is initialized to zero before the
        #"for image_name in image_names" loop to avoid overwriting character images in the "Dataset" folder.
        for i in range(len(chars_x_y_coordinates)):
            cropped_char = (text_image_threshold_240[chars_x_y_coordinates[i][0][1]:chars_x_y_coordinates[i][1][1],
                chars_x_y_coordinates[i][0][0]:chars_x_y_coordinates[i][1][0]])
            if not os.path.exists(cwd + "/Dataset/" + labels[i]):
                os.makedirs(cwd + "/Dataset/" + labels[i])
            file_path = cwd + "/Dataset/" + labels[i] + "/" + labels[i] + "-" + str(char_index) + ".jpg"
            cv2.imwrite(file_path, cropped_char)
            char_index +=1
        bar()
    #The code below removes the folder "to be deleted" and its contents.
    #Otherwise, the model would also train on this category that includes
    #all character rectangles labeled as "@" in the ".txt" files, which
    #represent mistakes in the typing of the dataset on the typewriter
    #(and not "#" overlaid with another character).
    if os.path.exists(cwd + "/Dataset/to be deleted/"):
        shutil.rmtree(cwd + "/Dataset/to be deleted/")
