from fastai.vision.all import *
import cv2
import os
import glob
from math import ceil


# For further information on how to implement convolutional neural networds with Fastai and
# Pytorch, please consult: "Howard, J., Gugger, S., & Safari, an O'Reilly Media Company.
# (2020). Deep learning for coders with fastai and PyTorch."

cwd = os.getcwd()
path =  os.path.join(cwd, "Dataset")

#If there is still a "to be deleted" subfolder within the "Dataset" folder,
#it will be deleted before determining the total number of label categories.
if os.path.exists(os.path.join(path, "to be deleted")):
    shutil.rmtree(os.path.join(path, "to be deleted"))


#The character images are not resized and while they are all of the same dimensions,
#these will vary according to the number of pixels in-between dots and the number
#of empty lines in-between lines of text. The size of the first JPEG image within
#the "Dataset" folder is retrieved automatically, in order to determine how many
#stride-2 convolutional neural network (CNN) layers will be required to reach a final
#layer size of 1, which is required to obtain a single output activation per image
#in order to train an image classification CNN model. The number of different
#characters to be subjected to OCR will also be stored in the variable "number_of_categories".
image_dimensions = None
number_of_categories = 0
for _, dirnames, filenames in os.walk(path):
    number_of_categories += len(dirnames)

for dirnames in os.walk(path):
    if image_dimensions == None:
        dataset_first_category_path = os.path.join(cwd, "Dataset", dirnames[1][0], "*.jpg")
    path_images = glob.glob(dataset_first_category_path)
    path_image = path_images[0]
    break
#The character image is loaded using the "cv2.imread" method.
character_image = cv2.imread(path_image)
#The height and width of the image is determined and stored in the
#"character_image_dimensions" list.
character_image_dimensions = [character_image.shape[0], character_image.shape[1]]

#The "while" loop will assemble the string that will be passed into the "eval()" method,
#in the "cnn_structure()" function. A stride-2 convolution is performed for each element within
#the "layers_list" list. That is to say, the layer size is doubled for every layer (except the
#first and last layers). For example, the third layer would be a 16x32 map of activations,
#hence the "if number_of_layers > 1" condition required to effect the stride-2 convolution.
#The "while" loop divides the "x" and "y" dimensions of the images by two at every layer,
#rounded up, until a layer size of 1x1 is obtained, indicating a single output activation layer.
dimension_x = character_image_dimensions[1]
dimension_y = character_image_dimensions[0]
number_of_layers = 0
layers_list = [[1,8], [8, 16]]
while dimension_x != 1 or dimension_y != 1:
    dimension_x = ceil(dimension_x/2)
    dimension_y = ceil(dimension_y/2)
    if number_of_layers > 1:
        layers_list.append([2*layers_list[-1][0], 2*layers_list[-1][1]])
    number_of_layers += 1

#The string is assembled by passing in the numbers within every layer, with
#the exception of the last layer before the flattening, which only contains
#a single output activation per image.
cnn_layers_structure = "sequential("
for i in range(len(layers_list)):
    cnn_layers_structure += ("conv(" + str(layers_list[i][0]) +
    ", " + str(layers_list[i][1]) + "), ")

cnn_layers_structure += ("conv(" + str(2*layers_list[i][0]) + ", " +
str(number_of_categories) + ", activation_function = False), Flatten())")

def get_dataloaders(batch_size=64):
    return DataBlock(
    blocks = (ImageBlock(cls=PILImageBW), CategoryBlock),
            get_items = get_image_files, splitter = RandomSplitter(valid_pct=0.2, seed = 42),
            get_y = parent_label,
            batch_tfms = Normalize()
            ).dataloaders(path, bs=batch_size)

def cnn_structure():
    #The string generated above is passed into the eval() method.
    return eval(cnn_layers_structure)

def fit(epochs=5, learning_rate = 0.003):
    learn = Learner(get_dataloaders(), cnn_structure(), loss_func=F.cross_entropy,
                    metrics = accuracy, cbs=ActivationStats(with_hist=True))
    learn.fit_one_cycle(epochs, learning_rate)
    return learn

def conv(number_inputs, number_features, kernel_size = 5, activation_function = True):
    layers = [nn.Conv2d(number_inputs, number_features, stride = 2,
        kernel_size = kernel_size, padding = kernel_size//2)]
    layers.append(nn.BatchNorm2d(number_features))
    if activation_function:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

learn = fit()

# #Uncomment the following lines of code allow to check the statistics of the model.
# print("")
# print(learn.summary())
# characters = DataBlock(blocks = (ImageBlock(cls=PILImageBW), CategoryBlock),
# get_items = get_image_files, splitter = RandomSplitter(valid_pct=0.2, seed = 42),
# get_y = parent_label,
# batch_tfms = Normalize())
# print("")
# print(characters.summary(path))
#
# #Uncomment the following lines of code allow to check batch normalization results
# dataloaders = get_dataloaders()
# x,y = dataloaders.one_batch()
# print(x.mean(dim=[0,2,3]), x.std(dim=[0,2,3]))

#Export the model
learn.export("handwriting_OCR_cnn_model")
