# TintypeText
This typewriter OCR application can convert JPEG typewritten text images into RTF documents, while removing typos for you!

![Image RTF basic mode](https://github.com/LPBeaulieu/Typewriter-OCR-TintypeText/blob/main/TintypeText%20basic%20rtf%20mode%20screenshot.jpg)
<h3 align="center">Tintype¶Text</h3>
<div align="center">
  
  [![License: AGPL-3.0](https://img.shields.io/badge/License-AGPLv3.0-brightgreen.svg)](https://github.com/LPBeaulieu/TintypeText/blob/main/LICENSE)
  [![GitHub last commit](https://img.shields.io/github/last-commit/LPBeaulieu/TintypeText)](https://github.com/LPBeaulieu/TintypeText)
  [![GitHub issues](https://img.shields.io/github/issues/LPBeaulieu/TintypeText)](https://github.com/LPBeaulieu/TintypeText)
  
</div>

---

<p align="left"> <b>Tintype¶Text</b> is a tool enabling you to convert scanned typewritten pages (in JPEG image format) into rich text format (RTF) 
  documents, complete with formatting elements such as text alignment, paragraphs, <u>underline</u>, <i>italics</i>, <b>bold</b> and <del>strikethrough</del>. </p>
<p align="left"> A neat functionality of <b>Tintype¶Text</b> is that the typos (wrong typewritten characters overlaid with a hashtag)
  automatically get filtered out, and do not appear in the final RTF text. This feature, when combined with erasable typewriter ink (see https://www.reddit.com/r/typewriters/comments/ujqxrh/how_to_make_erasable_typewriter_ink/) brings the typewriter well into the 21<sup>st</sup> century as a data entry tool!
  
  - You can get my <b>deep learning models</b> for both typewriters on which I developed the code on my Google Drive (<i>2021 
Royal Epoch</i> https://drive.google.com/drive/folders/1DUKqYf7wIkRAobC8fYPjum5gFOJqJurv?usp=sharing and <i>1968 Olivetti Underwood Lettera 33</i> https://drive.google.com/drive/folders/1sykG3zUfr8RJVbk59ClnzHjO3qgkXTmF?usp=sharing), where the datasets and other useful information to build your own datasets may be found. 
- The code showcased in this github page is the one that was used to generate a model with <b>99.93% optical character recognition (OCR) accuracy</b> with the 2021 Royal Epoch typewriter, which is in production and commercially available (I'm not affiliated with them, no worries).
- The generalizability of the model trained on a 2021 Royal Epoch typewriter was assessed on another unit of the same model (2019 Royal Epoch typewriter), with a text over 6,000 characters long. It gave an OCR accuracy of 99.22%, thus demonstrating that deep learning models trained with <b>Tintype¶Text</b> could be used with other typewriters of the same model (albeit with somewhat lower accuracy).
  
    <br> 
</p>

## 📝 Table of Contents
- [Dependencies / Limitations](#limitations)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## ⛓️ Dependencies / Limitations <a name = "limitations"></a>
- This Python project relies on the Fastai deep learning library (https://docs.fast.ai/) to generate a convoluted neural network 
  deep learning model, which allows for typewriter optical character recognition (OCR). It also needs OpenCV to perform image segmentation 
  (to crop the individual characters in the typewritten page images).
  
- A deep learning model trained on a specific typewriter is unlikely to generalize well to other typewriter brands, which may use different 
  typesets and character spacing. It is therefore preferable to train a model on your own typewriter.
- For best results, the typewritten text should be <b>double spaced</b> to avoid segmentation mistakes or omissions and the 8 1/2" x 11" typewritten pages should be <b>scanned at a resolution of 600 dpi</b>, as this resolution was used when writing the code.
- Every typewritten line should have <b>at least five adjoining letters</b> in order to be properly detected. Should a line only contain a word with 
  four or fewer letters, you could make up for the missing letters by using any character (other than "#") overlaid with a hashtag, which will 
  be interpreted by the code as an empty string, and will not impact the meaningful text on the line in the final rich text format (RTF) document.
- The <b>hashtag character is reserved</b> for designating typos, as a hyphen or equal sign overlaid with a hashtag are very similar to a hashtag 
  character by itself and would lead to OCR accuracy loss if it were used as a regular character.
- The <b>"@" symbol is reserved</b> to designate characters that are to be deleted (see description below) and should not be used on your typewriter, if it has such a type slug. 
- It should be noted that one of the typewriters with which the code was developed  (1968 Olivetti Underwood Lettera 33) doesn’t have specific type slugs for numbers zero (0) and one (1). After the OCR step, the Python code will interpret whether the surrounding characters are also digits 
  and assign the values to instances of uppercase “O” and lowercase “L” accordingly. It also converts the uppercase “O” into zero if it is 
  in one of the closing RTF formatting commands (e.g. \iO is changed to \i0). Even if your typewriter has type slugs for zero and one, make sure that they are very distinct in appearance from the uppercase “O” and lowercase “L” in order to ensure good OCR accuracy. Otherwise, just use the letters instead. Also, the <b>equal sign</b> on the typewriter is interpreted as a <b>backslash</b> if it is followed by a letter or an RTF escape (\\' (ASCII rtf character escape), \\- (hyphenation point) or \\_ (nonbreaking hyphen)), which is useful in RTF commands and escape codes. For an in-depth explanation of all the most common RTF commands and escapes, please consult: https://www.oreilly.com/library/view/rtf-pocket-guide/9781449302047/ch01.html. 
- To keep things as simple as possible in the (default) <b>basic RTF mode</b> of the "get_predictions.py" code, the use of curly brackets "{}" is disabled and "=par" is changed for "\par\pard" after OCR ("=" is used as there are no backslashes on typewriters). This means that the paragraph-formatting attributes (such as centered alignment, "<i>qc</i>" in the first line of the image above) are returned to their default values automatically when a new paragraph is started by typing "=par" on the typewriter.
- In the <b>advanced RTF mode</b>, the use of two successive parentheses "(( and ))" is translated to curly braces "{ and }", respectively, in the "get_predictions.py" Python code. Also, "=par" is changed to "\par" in the advanced RTF mode (and not to "\par\pard" as in the basic RTF mode). This allows more flexibility and the use of the curly brackets already limits the scope of the RTF commands, so there is no need to have a "\pard" added automatically. The image below illustrates how to use the parentheses in RTF commands in the advanced RTF mode. 

![Image RTF advanced mode](https://github.com/LPBeaulieu/Typewriter-OCR-TintypeText/blob/main/TintypeText%20advanced%20rtf%20mode%20image.jpg)<hr>

- It is recommended to include a space between your text and the parentheses (single or double, see image above), to reduce segmentation issues due to staggered character rectangles. The Python code automatically removes these spaces (if present) in the final RTF document (see image above). 

Despite these issues, the code has successfully located characters (segmentation step) on lines with at least 5 successive letters with a success 
rate above 99.99% for the training/validation data consisting of over 25,000 characters. The only issue reported with the training/validation 
data was an omitted period. As for the OCR accuracy, it was consistently above 99.8% regardless of the hyperparameters investigated (other than kernel size), provided
a good-sized dataset is used for training. 


## 🏁 Getting Started <a name = "getting_started"></a>

The following instructions will be provided in great detail, as they are intended for a broad audience and will
allow to run a copy of <b>Tintype¶Text</b> on a local computer. Here is a link to an instructional video explaining the steps 1 through 8 described below: https://www.youtube.com/watch?v=FG9WUW6q3dI&list=PL8fAaOg_mhoEZkbQuRgs8MN-QSygAjdil&index=2.

The paths included in the code are formatted for Unix (Linux) operating systems (OS), so the following instructions 
are for Linux OS environments.

<b>Step 1</b>- Go to the command line in your working folder and install the <b>Atom</b> text editor to make editing the code easier:
```
sudo snap install atom --classic
```

<b>Step 2</b>- Create a virtual environment (called <i>env</i>) in your working folder:
```
python3 -m venv env
```

<b>Step 3</b>- Activate the <i>env</i> virtual environment <b>(you will need to do this step every time you use the Python code files)</b> 
in your working folder:
```
source env/bin/activate
```

<b>Step 4</b>- Install <b>PyTorch</b> (Required Fastai library to convert images into a format usable for deep learning) using the following command (or the equivalent command found at https://pytorch.org/get-started/locally/ suitable to your system):
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

<b>Step 5</b>- Install the <i>CPU-only</i> version of <b>Fastai</b> (Deep Learning Python library, the CPU-only version suffices for this application, as the character images are very small in size):
```
pip install fastai
```

<b>Step 6</b>- Install <b>OpenCV</b> (Python library for image segmentation):
```
pip install opencv-python
```

<b>Step 7</b>- Install <b>alive-Progress</b> (Python module for a progress bar displayed in command line):
```
pip install alive-progress
```

<b>Step 8</b>- Create the folder "OCR Raw Data" in your working folder:
```
mkdir "OCR Raw Data" 
```
<b>Step 9</b>- You're now ready to use <b>Tintype¶Text</b>! 🎉

## 🎈 Usage <a name="usage"></a>
There are four different Python code files that are to be run in sequence. You can skip ahead to file 4 ("get_predictions.py") if you will be using one of the models in the Google Drive links above. You can find instructions for every Python file in the TintypeText - Typewriter Optical Character Recognition (OCR) playlist on my YouTube channel: https://www.youtube.com/playlist?list=PL8fAaOg_mhoEZkbQuRgs8MN-QSygAjdil.<br><br>
<b>File 1: "create_rectangles.py"</b>- This Python code enables you to see the segmentation results (the green rectangles delimiting
the individual characters on the typewritten image) and then write a ".txt" file with the correct labels for each rectangle. The mapping
of every rectangle to a label will allow to generate a dataset of character images with their corresponding labels. The typewriter
page images overlaid with the character rectangles are stored in the "Page image files with rectangles" folder, which is created
automatically by the code.

You might need to <b>alter the values</b> of the variables "<b>character_width</b>" (default value of 55 pixels for 8 1/2" x 11" typewritten pages 
scanned at a resolution of 600 dpi) and "<b>spacer_between_characters</b>" (default value of 5 pixels), as your typewriter may have a different typeset than those of my typewriters (those two default parameters were suitable for both my <i>2021 Royal Epoch</i> and <i>1968 Olivetti Underwood Lettera 33</i> typewriters). Also, if your typewriter has a lot of ghosting (faint outline of the preceding character) or if the signal to noise ratio is elevated (because of high ink loading on the ribbon leading to lots of ink speckling on the page), the segmentation code might pick up the ghosting or noise as characters. As a result, you could then end up with staggered character rectangles. In the presence of dark typewritten text you should decrease the segmentation sensitivity (increase the number of non-white y pixels required for a given x coordinate in order for that x coordinate to be included in the segmentation). That is to say that on a fresh ribbon of ink, you should increase the value of 3 (illustrated below) to about 6 (results will vary based on your typewriter's signal to noise ratio) in the line 57 of "get_predictions.py" in order to avoid including unwanted noise in the character rectangles. 
```
x_pixels = np.where(line_image >= 3)[0] 
```
When your typewritten text gets fainter, change that digit back to 3 to make the segmentation more sensitive (to avoid omitting characters). These parameters ("character_width", "spacer_between_characters" and "line_image >= 3" should be adjusted in the same way in all the Python code files (except "train_model.py", where they are absent) to ensure consistent segmentation in all steps of the process.

![Image txt file processing](https://github.com/LPBeaulieu/TintypeText/blob/main/txt%20file%20example.jpg)<hr>
The image above illustrates the format of the ".txt" file listing all of the character rectangle labels. In the first line, you can note that four of the characters are labeled as "@", which maps to the category "to be deleted". The three letters (C, X and I) have significant ink splattering and will not be included in the training data, as they are not representative of these characters. The fourth "@" on the first line corresponds to an artifact (some noise was above the filtering threshold and was picked up as a character). We also do not want to include it in the training data. The "lesser than" symbol highlighted in yellow on line 11 in the ".txt" file corresponds to an "empty" rectangle, which is mapped to the "space" category in the "Dataset" folder. The very last line of the typewriter scan image contains two typos (two characters overlaid with a hashtag symbol). They are represented by a "~" symbol in the ".txt" file on line 19. All the other character rectangles are represented by their own characters in the ".txt" file. 
<br><br>
Importantly, <b>such ".txt" files should be created, modified and saved exclusively in basic text editors</b> (such as Text Editor in Ubuntu 20.04), as more elaborate word processors would include extra formatting information that would interfere with the correct mapping of the character rectangles to their labels in the ".txt" file.

<b>Furthermore, the ".txt" files in the "Training&Validation Data" folder must have identical names to their corresponding JPEG images (minus the file extensions).</b> For example, the file "my_text.txt" would contain the labels corresponding to the raw scanned typewritten page JPEG image (without the character rectangles) named "my_text.jpg". The presence of hyphens in the file name is only necessary for JPEG files intended for OCR predictions (see below, file 4 "get_predictions.py"), although you could include some hyphens in every file name just as well.

<br>
 <b>File 2: "create_dataset.py"</b>- This code will crop the individual characters in the same way as the "create_rectangles.py" code,
 and will then open the ".txt" file containing the labels in order to create the dataset. Each character image will be sorted in its
 label subfolder within the "Dataset" folder, which is created automatically by the code. <br><br>
 A good practice <b>when creating a dataset</b> is to make the ".txt" file and then run the "create_dataset.py" code <b>one page at a time</b> (only one JPEG image and its corresponding ".txt" file at a time in the "Training&Validation Data" folder) to validate that the labels in the ".txt" file line up with the character rectangles on the typewritten text image. Such a validation step involves opening every "Dataset" subfolder and ensuring that every image corresponds to its subfolder label (pro tip: select the icon display option in the folder in order to display the image thumbnails, which makes the validation a whole lot quicker). You will need to delete the "Dataset" folder in between every page, otherwise it will add the labels to the existing ones within the subfolders. This makes it more manageable to correct any mistakes in the writing of the ".txt" files. Of note, some of the spaces are picked up as characters and framed with rectangles. You need to label those spaces with a lesser-than sign ("<"). Here is the list of symbols present in the ".txt" files mapping to the different characters rectangles:
  
  - <b>"<"</b>: "blank" character rectangle, which corresponds to a space. These character images are stored in the "space" subfolder within the "Dataset" folder.
  - <b>"~"</b>: "typo" character rectangle (any character overlaid with "#"). These character images are stored in the "empty" subfolder within the "Dataset" folder. 
  - <b>"@"</b>: "to be deleted" character rectangle (any undesired artifact or typo that wasn't picked up while typing on the typewriter). The 
    "to be deleted" subfolder (within the "Dataset" folder) and all its contents is automatically deleted and the characters labeled with "@" in the ".txt" file will be absent
    from the dataset, to avoid training on this erroneous data.
  - All the other characters in the ".txt" files are the same as those that you typed on your typewriter. The character images are stored in subfolders within the "Dataset" folder bearing the character's name (e.g. "a" character images are stored in the subfolder named "a").
 
  <b>Once you're done validating</b> the individual ".txt" files, you can delete the "Dataset" folder once more, add <b>all of the ".txt" files along with their corresponding JPEG images</b> to the "Training&Validation Data" folder and run the "create_dataset.py" code to get your complete dataset! 
  
![Image folder tree structure](https://github.com/LPBeaulieu/TintypeText/blob/main/Folder%20tree%20structure%20image.jpg)<hr>
The image above shows the folder tree structure of your working folder (above), along with the label subfolders within the "Dataset" folder (below).
 
  <br><b>File 3: "train_model.py"</b>- This code will train a convoluted neural network deep learning model from the labeled character images 
  within the "Dataset" folder. It will also provide you with the accuracy of the model in making OCR predictions, which will be displayed
  in the command line for every epoch (run through the entire dataset). The default hypeparameters (number of epochs=3, batch size=64, 
  learning rate=0.005, kernel size=5) were optimal and consistently gave OCR accuracies above 99.8%, provided a good-sized dataset is used (above 25,000 characters).  
  In my experience with this project, varying the value of any hyperparameter other than the kernel size did not lead to significant variations in accuracy.
  As this is a simple deep learning task, the accuracy relies more heavily on having good quality segmentation and character images that 
  accurately reflect those that would be found in text. Ideally, some characters would be typed with a fresh typewriter ribbon and others with an old one,
  to yield character images of varying boldness, once again reflecting the irregularities normally observed when using a typewriter.
  
  When you obtain a model with good accuracy, you should rename it and do a backup of it along with the "Dataset" folder on which it was trained.
  If you do change the name of the model file, you also need to update its name in the line 174 of "get_predictions.py":
  ```
  learn = load_learner(cwd + '/your_model_name')
  ```
  <br><b>File 4: "get_predictions.py"</b>- This code will perform OCR on JPEG images of scanned typewritten text (at a resolution of 600 dpi)
  that you will place in the folder "OCR Raw Data". 
  
  <b>Please note that all of the JPEG file names in the "OCR Raw Data" folder must contain at least one hyphen ("-") in order for the code
  to properly create subfolders in the "OCR Predictions" folder. These subfolders will contain the rich text format (RTF) OCR conversion documents.</b> 
  
  The reason for this is that when you will scan a multi-page document in a multi-page scanner, you will provide your scanner with a file root name (e.g. "my_text-") and the scanner will number them automatically (e.g."my_text-.jpg", "my_text-0001.jpg", "my_text-0002.jpg", "my_text-"0003.jpg", etc.) and the code would then label the subfolder within the "OCR Predictions" folder as "my_text". The OCR prediction results for each page will be added in sequence to the "my_text.rtf" file within the "my_text" subfolder of the "OCR Predictions" folder. Should you ever want to repeat the OCR prediction for a set of JPEG images, it would then be important to remove the "my_text" subfolder before running the "get_predictions.py" code once more, in order to avoid appending more text to the existing "my_text.rtf" file.

If you changed the name of your deep learning model, or if you are using one of the models that I trained, you will to update the model name within the "get_predictions.py" code. That is to say that you will need to change "typewriter_OCR_cnn_model" for the name of your model in line 174 of "get_predictions.py":
               
```              
learn = load_learner(cwd + '/typewriter_OCR_cnn_model')
```
               
As mentioned above, since fresh typewriter ink ribbons lead to darker text and more ink speckling on the page, in the presence of dark typewritten text you should decrease the segmentation sensitivity (increase the number of non-white y pixels required for a given x coordinate in order for that x coordinate to be included in the segmentation). That is to say that on a fresh ribbon of ink, you should increase the value of 3 (illustrated below) to about 6 (results will vary based on your typewriter's signal to noise ratio) in the line 56 of "get_predictions.py" in order to avoid including unwanted noise in the character rectangles. 
```
x_pixels = np.where(line_image >= 3)[0] 
```
When your typewritten text gets fainter, change that digit back to 3 to make the segmentation more sensitive (to avoid omitting characters).

        
  <br><b>And that's it!</b> You're now ready to convert your typewritten manuscript into digital format! You can now type away at the cottage or in the park without worrying about your laptop's battery life 
  and still get your document polished up in digital form in the end! 🎉📖
  
  
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
