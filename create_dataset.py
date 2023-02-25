#***IMPORTANT NOTE!!***
#You cannot start your actual JPEG file names with "back",
#as the presence of the "back" prefix in the file name
#designetes whether the page is even or odd numbered, with the
#file names for even-numbered pages starting with "back".

import cv2
import os
import shutil
import re
from alive_progress import alive_bar
import numpy as np
import sys
import math


cwd = os.getcwd()

#The list "JPEG_file_names" is populated with the ".jpg" file names in
#the "Training&Validation Data" folder.
JPEG_file_names = ([file_name for file_name in sorted(os.listdir(os.path.join(cwd,
"Training&Validation Data"))) if file_name[-4:] == ".jpg"])

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

#This code obtains the individual character coordinates from the image files
#listed in the "JPEG_file_names" list and generates JPEG images with overlaid
#character rectangles, named after the original files, but with the added
#"with character rectangles" suffix.
with alive_bar(len(JPEG_file_names)) as bar:
    for i in range(len(JPEG_file_names)):

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
        text_image = cv2.imread(os.path.join(cwd, "Training&Validation Data", str(JPEG_file_names[i])))
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



        #Building a list of character labels (labels were obtained from a txt file named after the "JPEG_file_names[i]",
        #which itself was written based on the image with overlaping character rectangles generated above).
        #An important note to those who would like to train their own models is to generate and save the txt
        #files exclusively in WordPad or Text Editor and not a full-fledged word processor, which would insert
        #formatting information that would skew the character count.
        labels = []
        with open(os.path.join('Training&Validation Data', JPEG_file_names[i][:-4] + '.txt'), 'r', encoding="utf-8") as f:
            text = f.read()
            character_count = 0
            for char in text:
                #If the character in the txt file is a space or a line carriage ("\n"),
                #no label is appended to the list "labels".
                if text[character_count] == ' ' or text[character_count:character_count+1] == "\n":
                    character_count+=1
                    pass
                #If the character in the txt file is a "И" (Cyrillic Capital Letter I,
                #Python source code u"\u0418"), "space" is appended to the list "labels".
                elif text[character_count] == u"\u0418":
                    labels.append("space")
                    character_count+=1
                #If the character in the txt file is a "Б" (Cyrillic Capital Letter Be,
                #Python source code u"\u0411"), "empty" is appended to the list "labels".
                elif text[character_count] == u"\u0411":
                    labels.append("empty")
                    character_count+=1
                #If the character in the txt file is a "/" (Python source code in [u"\u002F", u"\u2215", u"\u2044"]),
                #"forward slash" is appended to the list "labels".
                elif text[character_count] in [u"\u002F", u"\u2215", u"\u2044"]:
                    labels.append("forward slash")
                    character_count+=1
                #If the character in the txt file is a "\" (Python source code in [u"\u005C", u"\uFF3C"]),
                #"backslash" is appended to the list "labels".
                elif text[character_count] in [u"\u005C", u"\uFF3C"]:
                    labels.append("backslash")
                    character_count+=1
                #If the character in the txt file is a "|" (Python source code u"\u007C"),
                #"pipe" is appended to the list "labels".
                elif text[character_count] == u"\u007C":
                    labels.append("pipe")
                    character_count+=1
                #If the character in the txt file is a "$" (Python source code in [u"\u0024", u"\uFF04", u"\uFE69"]),
                #"dollar sign" is appended to the list "labels".
                elif text[character_count] in [u"\u0024", u"\uFF04", u"\uFE69"]:
                    labels.append("dollar sign")
                    character_count+=1
                #If the character in the txt file is a "+" (Python source code in [u"\uFF0B", u"\u002B"]),
                #"plus sign" is appended to the list "labels".
                elif text[character_count] in [u"\uFF0B", u"\u002B"]:
                    labels.append("plus sign")
                    character_count+=1
                #If the character in the txt file is a "=" (Python source codein [u"\u003D", u"\uFF1D"]),
                #"equals sign" is appended to the list "labels".
                elif text[character_count] in [u"\u003D", u"\uFF1D"]:
                    labels.append("equals sign")
                    character_count+=1
                #If the character in the txt file is a "?" (Python source code in [u"\u003F", u"\uFF1F"]),
                #"question mark" is appended to the list "labels".
                elif text[character_count] in [u"\u003F", u"\uFF1F"]:
                    labels.append("question mark")
                    character_count+=1
                #If the character in the txt file is a "!" (Python source code u"\u0021"), "exclamation mark"
                #is appended to the list "labels".
                elif text[character_count] == u"\u0021":
                    labels.append("exclamation mark")
                    character_count+=1
                #If the character in the txt file is a ".", "period"
                #is appended to the list "labels".
                elif text[character_count] == '.':
                    labels.append("period")
                    character_count+=1
                #If the character in the txt file is a ":" (Python source code u"\u003A"), "colon"
                #is appended to the list "labels".
                elif text[character_count] == u"\u003A":
                    labels.append("colon")
                    character_count+=1
                #If the character in the txt file is a "@" (Python source code u"\u0040"), "at sign"
                #is appended to the list "labels".
                elif text[character_count] == u"\u0040":
                    labels.append("at sign")
                    character_count+=1
                #If the character in the txt file is a "`" (Python source code in [u"\u0060", u"\u0300", u"\uFF40"]),
                #"grave accent" is appended to the list "labels".
                elif text[character_count] in [u"\u0060", u"\u0300", u"\uFF40"]:
                    labels.append("grave accent")
                    character_count+=1
                #If the character in the txt file is a "'" (Python source code u"\u0027"), "single quote"
                #is appended to the list "labels".
                elif text[character_count] == u"\u0027":
                    labels.append("single quote")
                    character_count+=1
                #If the character in the txt file is a '"' (Python source code u"\u0022"), "double quote"
                #is appended to the list "labels".
                elif text[character_count] == u"\u0022":
                    labels.append("double quote")
                    character_count+=1
                #If the character in the txt file is a '#' (Python source code in [u"\u0023", u"\uFF03"]), "hashtag"
                #is appended to the list "labels".
                elif text[character_count] in [u"\u0023", u"\uFF03"]:
                    labels.append("hashtag")
                    character_count+=1
                #If the character in the txt file is a '<' (Python source code in [u"\u003C", u"\uFF1C"]),
                #"lesser-than sign" is appended to the list "labels".
                elif text[character_count] in [u"\u003C", u"\uFF1C"]:
                    labels.append("lesser-than sign")
                    character_count+=1
                #If the character in the txt file is a '>' (Python source code in [u"\u003E", u"\uFF1E", u"\uFE65"]),
                #"greater-than sign" is appended to the list "labels".
                elif text[character_count] in [u"\u003E", u"\uFF1E", u"\uFE65"]:
                    labels.append("greater-than sign")
                    character_count+=1
                #If the character in the txt file is a '*' (Python source code in [u"\u002A", u"\u2217"]), "asterisk"
                #is appended to the list "labels".
                elif text[character_count] in [u"\u002A", u"\u2217"]:
                    labels.append("asterisk")
                    character_count+=1
                #If the character in the txt file is a '%' (Python source code in [u"\u0025", u"\uFE6A", u"\uFF05"]),
                #"percent" is appended to the list "labels".
                elif text[character_count] in [u"\u0025", u"\uFE6A", u"\uFF05"]:
                    labels.append("percent")
                    character_count+=1
                #If the character in the txt file is a '&' (Python source code in [u"\uFF06", u"\u0026", u"\uFE60"]),
                #"ampersand" is appended to the list "labels".
                elif text[character_count] in [u"\uFF06", u"\u0026", u"\uFE60"]:
                    labels.append("ampersand")
                    character_count+=1
                #If the character in the txt file is a '{' (Python source code u"\u007B"), "left curly bracket"
                #is appended to the list "labels".
                elif text[character_count] == u"\u007B":
                    labels.append("left curly bracket")
                    character_count+=1
                #If the character in the txt file is a '}' (Python source code in [u"\u007D", u"\uFF5D"]),
                #"right curly bracket" is appended to the list "labels".
                elif text[character_count] in [u"\u007D", u"\uFF5D"]:
                    labels.append("right curly bracket")
                    character_count+=1
                #If the character in the txt file is a "Д" (Cyrillic Capital Letter De,
                #Python source code u"\u0414"), "to be deleted" is appended to the list "labels".
                elif text[character_count] == u"\u0414":
                    labels.append("to be deleted")
                    character_count+=1
                #Other characters get appended to the list.
                else:
                    labels.append(text[character_count])
                    character_count+=1

        #This "for" loop is needed only to determine how many
        #squares are found on the actual page ("page_character_index").
        #This needs to be done before repeating it below, as the user
        #needs to know how many characters are missing (if applicable),
        #in the "labels" list derived from the TXT file, before actually
        #cropping the characters. Otherwise, the user would only get an
        #error.
        page_character_index = 0
        for j in range(len(chars_x_y_coordinates)):
            for k in range(len(chars_x_y_coordinates[j])):
                page_character_index += 1

        #As an added quality control step, the length of the list of character coordinates and their labels
        #are printed on screen. The two lengths should be equivalent, as they both refer to the same characters.
        print("Currently working on file: " + os.path.basename(JPEG_file_names[i]))
        print("Length of list 'chars_x_y_coordinates': " + str(page_character_index))
        print("Length of list 'labels': " + str(len(labels)))
        print("")

        page_character_index = 0
        #Generating the individual character images based on their x and y coordinates (from chars_x_y_coordinates),
        #every image being placed in a folder corresponding to its label (labels were obtained from a txt file
        #named after the "JPEG_file_names[i]"). The counter "char_index" is initialized to zero before the
        #"for i in range(len(JPEG_file_names))" loop to avoid overwriting character images in the "Dataset" folder.

        for j in range(len(chars_x_y_coordinates)):
            for k in range(len(chars_x_y_coordinates[j])):
                cropped_char = (text_image_gray[chars_x_y_coordinates[j][k][0][1]:
                chars_x_y_coordinates[j][k][1][1], chars_x_y_coordinates[j][k][0][0]:
                chars_x_y_coordinates[j][k][1][0]])

                if not os.path.exists(os.path.join(cwd, "Dataset", labels[page_character_index])):
                    os.makedirs(os.path.join(cwd, "Dataset", labels[page_character_index]))
                file_path = os.path.join(cwd,  "Dataset", labels[page_character_index],
                labels[page_character_index] + "-" + str(character_index) + ".jpg")
                cv2.imwrite(file_path, cropped_char)
                page_character_index += 1
                character_index += 1

        bar()
#The code below removes the folder "to be deleted" and its contents.
#Otherwise, the model would also train on this category that includes
#all character rectangles labeled as "@" in the ".txt" files, which
#represent mistakes in the typing of the dataset on the typewriter
#(and not "#" overlaid with another character).
if os.path.exists(os.path.join(cwd, "Dataset", "to be deleted")):
    shutil.rmtree(os.path.join(cwd, "Dataset", "to be deleted"))
