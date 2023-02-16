# ScriptReader
This handwriting OCR application can convert JPEG handwritten text images into RTF documents, while removing typos for you!

![Image RTF basic mode](https://github.com/LPBeaulieu/Typewriter-OCR-TintypeText/blob/main/TintypeText%20basic%20rtf%20mode%20screenshot.jpg)
<h3 align="center">ScriptReader</h3>
<div align="center">
  
  [![License: AGPL-3.0](https://img.shields.io/badge/License-AGPLv3.0-brightgreen.svg)](https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader/blob/main/LICENSE)
  [![GitHub last commit](https://img.shields.io/github/last-commit/LPBeaulieu/Handwriting-OCR-ScriptReader)](https://github.com/LPBeaulieu/Handwriting-OCR-ScriptReader)
  [![GitHub issues](https://img.shields.io/github/issues/LPBeaulieu/Handwriting-OCR-ScriptReader)](https://github.com/LPBeaulieu/TintypeText)
  
</div>

---

<p align="left"> <b>ScriptReader</b> is a tool enabling you to convert scanned handwritten pages (in JPEG image format) into rich text format (RTF) 
  documents, complete with formatting elements such as text alignment, paragraphs, <u>underline</u>, <i>italics</i>, <b>bold</b> and <del>strikethrough</del>. </p>
<p align="left"> A neat functionality of <b>ScriptReader</b> is that the typos (grid cells containing errors that are filled in with ink)
  automatically get filtered out, and do not appear in the final RTF text. This feature, when combined with erasable biodegradable fountain pen ink (see https://www.linkedin.com/feed/update/urn:li:activity:7027469654725947392/) further enhances the usefulness of the notebooks you print with PrintANotebook (see https://github.com/LPBeaulieu/Notebook-Maker-PrintANotebook)!
<br> 
</p>

## 📝 Table of Contents
- [Dependencies / Limitations](#limitations)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## ⛓️ Dependencies / Limitations <a name = "limitations"></a>
- This Python project relies on the Fastai deep learning library (https://docs.fast.ai/) to generate a convolutional neural network 
  deep learning model, which allows for typewriter optical character recognition (OCR). It also needs OpenCV to perform image segmentation 
  (to crop the individual characters in the typewritten page images) and glob2 to automatically retrieve the cropped character image size.
  
- A deep learning model trained with a specific handwriting is unlikely to generalize well to other handwritings. Also, I would advise you to keep your dataset private, as it would in theory be possible to reverse engineer it in order to generate text with your handwriting. <b>For this reason, I encourage you to use another handwriting than your official handwriting, as an added precaution.</b>
- The Scriptreader pages from PrintANotebook need to be used, and the individual letters need to be written within the horizontal boundaries of a given dot grid square cell (comprised of four dots). The segmentation code allows plenty of space for ascenders and descenders, however. The handwritten pages should be <b>scanned at a resolution of 300 dpi</b>, with the US Letter page size setting and the text facing the top of the page, as the black squares will be used to automatically align the pages. You should refrain from writing near the dark squares to allow for the alignment to be unimpeded by any artifacts.  

- Make sure that all of your characters are very distinct from one another. I suggest using bars or dots in the zeros and writings the "1"
the way you see it displayed on screen, so that it isn't confused with a lowercase "l".

- Also, the code is compatible with RTF commands (see example above) so you will need to train the model to recognize hand-drawn backslashes as well if you wish to include formatting elements such as tabs, bold, new lines and paragraphs, for instance. For an in-depth explanation of all the most common RTF commands and escapes, please consult: https://www.oreilly.com/library/view/rtf-pocket-guide/9781449302047/ch01.html. 

- To keep things as simple as possible in the (default) <b>basic RTF mode</b> of the "get_predictions.py" code, the use of curly brackets "{}" is disabled and "\par" is changed for "\par\pard\tab" after OCR. This means that the paragraph-formatting attributes (such as centered alignment, "<i>qc</i>") are returned to their default values, and a tab is included automatically when a new paragraph is started by writing "\par". The <b>advanced RTF mode</b>, just interprets the RTF commands as you write them.

My preliminary tests with 17 000 characters of training data (29 pages of cursive handwriting on the ScriptReader pages, with 0.13 inch dot spacing and double line spacing) gave me an OCR accuracy above 98%, and I am still adding more data to the model, so there is room for improvement.


## 🏁 Getting Started <a name = "getting_started"></a>

The following instructions will be provided in great detail, as they are intended for a broad audience and will
allow to run a copy of <b>ScriptReader</b> on a local computer.

The instructions below are for Windows operating systems, but the code should run nicely on Linux and Mac-OS as well.

<b>Step 1</b>- Hold the "Shift" key while right-clicking in your working folder and select "Open PowerShell window here" to access the PowerShell in your working folder. Then, install the <b>Atom</b> text editor to make editing the code easier:
```
sudo snap install atom --classic
```

<b>Step 2</b>- Install <b>PyTorch</b> (Required Fastai library to convert images into a format usable for deep learning) using the following command (or the equivalent command found at https://pytorch.org/get-started/locally/ suitable to your system):
```
pip3 install torch torchvision torchaudio
```

<b>Step 3</b>- Install the <i>CPU-only</i> version of <b>Fastai</b> (Deep Learning Python library, the CPU-only version suffices for this application, as the character images are very small in size):
```
py -m pip install fastai
```

<b>Step 4</b>- Install <b>OpenCV</b> (Python library for image segmentation):
```
py -m pip install opencv-python
```

<b>Step 5</b>- Install <b>alive-Progress</b> (Python module for a progress bar displayed in command line):
```
py -m pip install alive-progress
```

<b>Step 6</b>- Install <b>glob</b> (Python module used to automatically retrieve the cropped character image pixel size):
```
py -m pip install glob2
```

<b>Step 7</b>- Create the folder "OCR Raw Data" in your working folder:
```
mkdir "OCR Raw Data" 
```
<b>Step 9</b>- You're now ready to use <b>ScriptReader</b>! 🎉

## 🎈 Usage <a name="usage"></a>
There are four different Python code files that are to be run in sequence. You can find instructions for every Python file in the ScriptReader playlist on my YouTube channel: **The link will be posted when the videos are uploaded**.<br><br>
<b>File 1: "create_rectangles.py"</b>- This Python code enables you to see the segmentation results (the green rectangles delimiting
the individual characters on the handwritten scanned image) and then write a ".txt" file with the correct labels for each rectangle. The mapping
of every rectangle to a label will allow to generate a dataset of character images with their corresponding labels. The handwritten
page images overlaid with the character rectangles are stored in the "Page image files with rectangles" folder, which is created
automatically by the code.

![Image txt file processing](https://github.com/LPBeaulieu/TintypeText/blob/main/txt%20file%20example.jpg)<hr>
The image above illustrates the format of the ".txt" file listing all of the character rectangle labels. Any spaces (empty dot grid cells) on your scanned handwritten page need to be represented by the Cyrillic Capital Letter I ("И"). Furthermore, any typos on your training data, which are erroneous handwritten letters of which the dot grid cell was later darkened will need to be reported with the Cyrillic Capital Letter Be ("Б"). Finally, if there are artifacts on the page or if you wrote a character out of frame of a given dot grid cell, you need to designate it with the Cyrillic Capital Letter De ("Д"). All the other character rectangles are represented by their own characters in the ".txt" file. 
<br><br>
Importantly, <b>such ".txt" files should be created, modified and saved exclusively in basic text editors</b> (such as Notepad on Windows or Text Editor on Ubuntu), as more elaborate word processors would include extra formatting information that would interfere with the correct mapping of the character rectangles to their labels in the ".txt" file.

<b>Furthermore, the ".txt" files in the "Training&Validation Data" folder must have identical names to their corresponding JPEG images (minus the file extensions).</b> For example, the file "my_text.txt" would contain the labels corresponding to the raw scanned handwritten page JPEG image (without the character rectangles) named "my_text.jpg". The presence of hyphens in the file name is only necessary for JPEG files intended for OCR predictions (see below, file 4 "get_predictions.py"), although you could include some hyphens in every file name just as well.

<br>
 <b>File 2: "create_dataset.py"</b>- This code will crop the individual characters in the same way as the "create_rectangles.py" code,
 and will then open the ".txt" file containing the labels in order to create the dataset. Each character image will be sorted in its
 label subfolder within the "Dataset" folder, which is created automatically by the code. <br><br>
 A good practice <b>when creating a dataset</b> is to make the ".txt" file and then run the "create_dataset.py" code <b>one page at a time</b> (only one JPEG image and its corresponding ".txt" file at a time in the "Training&Validation Data" folder) to validate that the labels in the ".txt" file line up with the character rectangles on the handwritten text image. Such a validation step involves opening every "Dataset" subfolder and ensuring that every image corresponds to its subfolder label (pro tip: select the icon display option in the folder in order to display the image thumbnails, which makes the validation a whole lot quicker). You will need to delete the "Dataset" folder in between every page, otherwise it will add the labels to the existing ones within the subfolders. This makes it more manageable to correct any mistakes in the writing of the ".txt" files. Of note, the spaces are picked up as actual characters and framed with rectangles. You need to label those spaces with a "И" symbol. Here is the list of symbols present in the ".txt" files mapping to the different characters rectangles:
  
  - <b>"И"</b>: "blank" character rectangle, which corresponds to a space. These character images are stored in the "space" subfolder within the "Dataset" folder.
  - <b>"Б"</b>: "typo" character rectangle (any character of which the square dot grid cell has been completely darkened). These character images are stored in the "empty" subfolder within the "Dataset" folder. 
  - <b>"Д"</b>: "to be deleted" character rectangle (any undesired artifact or letter that is out of frame with the dot grid cell). The 
    "to be deleted" subfolder (within the "Dataset" folder) and all its contents is automatically deleted and the characters labeled with "Д" in the ".txt" file will be absent from the dataset, to avoid training on this erroneous data.
  - All the other characters in the ".txt" files are the same as those that you have written by hand. The character images are stored in subfolders within the "Dataset" folder bearing the character's name (e.g. "a" character images are stored in the subfolder named "a").
 
  <b>Once you're done validating</b> the individual ".txt" files, you can delete the "Dataset" folder once more, add <b>all of the ".txt" files along with their corresponding JPEG images</b> to the "Training&Validation Data" folder and run the "create_dataset.py" code to get your complete dataset! 
  
![Image folder tree structure](https://github.com/LPBeaulieu/TintypeText/blob/main/Folder%20tree%20structure%20image.jpg)<hr>
The image above shows the folder tree structure of your working folder (above), along with the label subfolders within the "Dataset" folder (below).
 
  <br><b>File 3: "train_model.py"</b>- This code will train a convolutional neural network deep learning model from the labeled character images 
  within the "Dataset" folder. It will also provide you with the accuracy of the model in making OCR predictions, which will be displayed
  in the command line for every epoch (run through the entire dataset). The default hypeparameters (number of epochs=3, batch size=64, 
  learning rate=0.005, kernel size=5) were optimal and consistently gave OCR accuracies above 98%, provided a good-sized dataset is used (mine was above 17,000 characters). As this is a simple deep learning task, the accuracy relies more heavily on having good quality segmentation and character images that accurately reflect those that would be found in text. When you obtain a model with good accuracy, you should rename it and do a backup of it along with the "Dataset" folder on which it was trained. If you do change the name of the model file, you also need to update its name in the line 95 of "get_predictions.py":
  ```
  learn = load_learner('handwriting_OCR_cnn_model')
  ```
  <br><b>File 4: "get_predictions.py"</b>- This code will perform OCR on JPEG images of scanned handwritten text (at a resolution of 300 dpi and with the US Letter page size setting) that you will place in the folder "OCR Raw Data". 
  
  <b>Please note that all of the JPEG file names in the "OCR Raw Data" folder must contain at least one hyphen ("-") in order for the code
  to properly create subfolders in the "OCR Predictions" folder. These subfolders will contain the rich text format (RTF) OCR conversion documents.</b> 
  
  The reason for this is that when you will scan a multi-page document in a multi-page scanner, you will provide your scanner with a file root name (e.g. "my_text-") and the scanner will number them automatically (e.g."my_text-.jpg", "my_text-0001.jpg", "my_text-0002.jpg", "my_text-"0003.jpg", etc.) and the code would then label the subfolder within the "OCR Predictions" folder as "my_text". The OCR prediction results for each page will be added in sequence to the "my_text.rtf" file within the "my_text" subfolder of the "OCR Predictions" folder. Should you ever want to repeat the OCR prediction for a set of JPEG images, it would then be important to remove the "my_text" subfolder before running the "get_predictions.py" code once more, in order to avoid appending more text to the existing "my_text.rtf" file.
              
        
  <br><b>Well there you have it!</b> You're now ready to convert your handwritten text into digital format! You can now write at the cottage or in the park without worrying about your laptop's battery life and still get your document polished up in digital form in the end! 🎉📖
  
  
## ✍️ Authors <a name = "author"></a>
- 👋 Hi, I’m Louis-Philippe!
- 👀 I’m interested in natural language processing (NLP) and anything to do with words, really! 📝
- 🌱 I’m currently reading about deep learning (and reviewing the underlying math involved in coding such applications 🧮😕)
- 📫 How to reach me: By e-mail! LPBeaulieu@gmail.com 💻


## 🎉 Acknowledgments <a name = "acknowledgments"></a>
- Hat tip to [@kylelobo](https://github.com/kylelobo) for the GitHub README template!




<!---
LPBeaulieu/LPBeaulieu is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
