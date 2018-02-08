# **Problem Statement :**
- ### `To extract  the registration number of a car entering inside a parking lot.`

# **Softwares and Technology used :**
- OpenCV 3
- Python 3
- Tensor Flow
- MongoDB
- Anaconda package manager

# Refer to "How to run the program.txt" for instructions on running the code.

# INITIAL BACKGROUND

- ANPR is an image-processing innovation which is used to perceive vehicles by their tags. This expertise is ahead of time ubiquity in security and traffic installation. Tag Recognition System is an application of PC vision. PC vision is a technique for using a PC to take out abnormal state information from a digital image. The useless homogeny among various tags for example, its dimension and the outline of the License Plate. The ALPR system consists of following steps:-
  - Vehicle image capture.
  - Preprocessing.
  - Number plate extraction.
  - Character segmentation.
  - Character recognition.





- The ALPR system works in these strides, the initial step is the location of the vehicle and capturing a vehicle image of front or back perspective of the vehicle, the second step is the localization of Number Plate and then extraction of vehicle Number Plate is an image. The final stride use image segmentation strategy, for the segmentation a few techniques neural network, mathematical morphology, color analysis and histogram analysis. Segmentation is for individual character recognition. Optical Character Recognition (OCR) is one of the strategies to perceive the every character with the assistance of database stored for separate alphanumeric character.

# TESTING REPORT

- Let  ‘correct’  =  (number of license plates correctly detected)
	Let  ‘total images’ = total number of license plates.

	Total images = 480

	Accuracy = (correct * 100) / Total images
- Avg. image time = (total time)/Total images
- All testing is done on  Processor: Inter(R) Core(TM)
i5 -6200 CPU @ 2.30 Ghz
Memory: 8GB DDR - 4
- CNN Classifier is trained on GPU: Nvidia GeForce GTX 960
Graphics Memory : 4 GB
Processor: Inter(R) Core(TM)  i7 -4710 HQ CPU @ 3.2 Ghz
Memory : 8 GBDDR-4

# **Approach used and Implemented :**

- In this project we start with the process.py file that asks to input  the name of the video file for which we want the prediction.
This then breaks the video into frames and stores them in a folder name ‘data’.
The we run our send each frame to the main.py which will return the predicted image and the cropped license plate from the frame.
We have displayed a test run on a car image shown below.

- ## 1) Loading the Image :
- ## 2) PreProcess the input image :
  - This step in done in the Preprocess.py file.
	As most of the opencv function require the input iamge to be in the grayscale and greyscale images are easy for computation so we convert the input image to greyscale.
- ## 3) Plate detection :
  - Now we have a grayscale image and a thresholded image. We send them to the DetectPlates.py file. Next we apply the findcountour function of opencv to detection all the boundaries in the thresholded image. This step is to extract the characters from the image so that they can be recognized.
- ## 4) Character Segmentation :
	- This phase is done in the DetectChar.py file.
	Here we start with each possible plates.
	We crop the plate from input image and resize it to 1.6 times height, 1.6 times width. The we again apply the preprocessing operations on the plate image.
- ## Character recognition :
	- This part is done in the train_detect.py file.
  For the purpose of recognition of the cropped character we use a Trained Convolutional Neural Network classifier.
  - The classifier is built using the python keras module.
  - The classifier is trainied with 47605 images constituting images 36 classes ranging from 0-9 and A-Z.
  - The classifier is tested with 1292 images constituting images 36 classes ranging from 0-9 and A-Z
  - It takes input 64x64 input image in the first convolution layer.
  - The Relu activation function in the convolution layer.
  - The softmax function is used in the final prediction layer.The classifier uses the Root Mean square optimizer with the starting learning rate as 0.001 which gradually decreases by 0.005 after each epoch. We trained the classifier for 5 epoches.

- ## 6.) Prediction :
	- With this we have the predicted number plate and the cropped image of the number plate from the frame. Next we store the number_plate predicted and the image  in a dictionary with key as the predicted number_plate. If it is already in the dictionary then we increment its count by 1.In the end we predict the number_plate with the maximum frequency as the number on the number plate of the car in the video. We write the cropped image of the number plate in the same directirary with name as video name .jpf.
- ## 7.) Storing the Results in MongoDB Database :
  - After we have got the prediction from our main function we then store the Name plate, Cropped image, time of prediction, frequency of the number plate in the dictionary, execution time and name of the video in the pymongo database.
- ## Application of ALPR :
  - Automatic license-plate recognition (ALPR) is a technology that uses optical character recognition on images to read vehicle registration plates. It can use existing closed-circuit television, road-rule enforcement cameras, or cameras specifically designed for the task. ALPR can be used by police forces around the world for law enforcement purposes, including to check if a vehicle is registered or licensed. It is also used for electronic toll collection on pay-per-use roads and as a method of cataloguing the movements of traffic, for example by highways agencies.
  - Automated License Plate Recognition has many uses including:
    - Recovering stolen cars.
    - Identifying drivers with an open warrant for arrest.
    - Catching speeders by comparing the average time it takes to get from stationary camera A to stationary camera B.
    - Determining what cars do and do not belong in a parking garage.
    - Expediting parking by eliminating the need for human confirmation of parking passes.
