'''
The main script that contains all the functions to end to end process an image and give the results.
@input: Image of the car with number plate visible
@output:List of number plates with their respective text.
'''

# Importing the Libraries
import cv2
import numpy as np
import math
import random
import Preprocess
import PossibleChar
import PossiblePlate
import os
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import ops as utils_ops
import argparse

# module level variables ##########################################################################
# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # other constants
MIN_NUMBER_OF_MATCHING_CHARS = 4

RESIZED_CHAR_IMAGE_WIDTH = 64
RESIZED_CHAR_IMAGE_HEIGHT = 64

MIN_CONTOUR_AREA = 100
model = load_model('models/weights-improvement-03-0.93.hdf5')


###################################################################################################


def loadCNNClassifier():
    # compile the character-digit detection model
    model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return True
###################################################################################################

def detectCharsInPlates(listOfPossiblePlates,save_intermediate,output_folder,showSteps):
    '''
    This function processes the list of number plates and returns a lsit of processed number plates
    with their text.
    @input: List of images of number plates
    @output:A list of number plates with all the information encapsulated
    '''

    # Initialize the varaibles
    intPlateCounter = 0
    imgContours = None
    contours = []


    if len(listOfPossiblePlates) == 0:          # if list of possible plates is empty
        l = []
        l = l.append(listOfPossiblePlates)
        print('No Plates found')
        return l        # return
    # end if


    # at this point we can be sure the list of possible plates has at least one plate
    listOfPossiblePlates_refined = []
    for possiblePlate in listOfPossiblePlates:          # for each possible plate, this is a big for loop that takes up most of the function
        #possiblePlate.imgPlate = cv2.fastNlMeansDenoisingColored(possiblePlate.imgPlate,None,15,15,7,21)
        #possiblePlate.imgPlate = cv2.equalizeHist(possiblePlate.imgPlate)

        # preprocess to get grayscale and threshold images
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate,save_intermediate,output_folder,showSteps)  

        if showSteps == True: # show steps ###################################################
            cv2.imshow("imgPlate", possiblePlate.imgPlate)
            cv2.imshow("imgGrayscale", possiblePlate.imgGrayscale)
            cv2.imshow("imgThresh", possiblePlate.imgThresh)
            cv2.waitKey(0)
                        # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6,interpolation=cv2.INTER_LINEAR)

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # This clears the image more removing all the unknown noise from it.
        if showSteps == True: # show steps ###################################################
            cv2.imshow("imgThresh_gray_remover", possiblePlate.imgThresh)
            cv2.waitKey(0)

        if save_intermediate == True: # show steps ###################################################
            cv2.imwrite("%s/imgThresh_gray_remover.png"%(output_folder), possiblePlate.imgThresh)
            
        # end if # show steps #####################################################################

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars (without 
        # comparison to other chars yet)
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if showSteps == True or save_intermediate == True:
            height, width = possiblePlate.imgThresh.shape
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]                                         # clear the contours list

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (255,255,255))
            #print('These are the possible characters in the plate :')

        if showSteps == True: # show steps ###################################################
            cv2.imshow("Possible_chars_in_plate", imgContours)
            cv2.waitKey(0)

        if save_intermediate == True: # show steps ###################################################
            cv2.imwrite("%s/Possible_chars_in_plate.png"%(output_folder), imgContours)
            
        # end if # show steps #####################################################################


        # given a list of all possible chars, find groups of matching chars within the plate
        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        if (len(listOfListsOfMatchingCharsInPlate) == 0):            # if no groups of matching chars were found in the plate
            #print('\nNo matching characters found:')
            if showSteps == True: # show steps ###############################################
                print("chars found in plate number " + str(intPlateCounter) + " = (none), click on any image and press a key to continue . . .")
                intPlateCounter = intPlateCounter + 1

                cv2.destroyAllWindows()
            # end if # show steps #################################################################

            possiblePlate.strChars = ""
            continue                        # go back to top of for loop
        # end if

        if showSteps == True or save_intermediate == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
        
        if showSteps == True:
            cv2.imshow("A_Complete_plate", imgContours)
            cv2.waitKey(0)
        # end if # show steps #####################################################################


        if save_intermediate == True: # show steps ###################################################
            cv2.imwrite("%s/A_Complete_plate.png"%(output_folder), imgContours)
            


         # within each list of matching chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):                             
            # sort chars from left to right
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)        

            # and remove inner overlapping chars
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])              
        # end for

        if showSteps == True or save_intermediate == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # end for

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # end for
        
        if showSteps == True:
            cv2.imshow("Remove_Overlapping", imgContours)
            cv2.waitKey(0)

        if save_intermediate == True: # show steps ###################################################
            cv2.imwrite("%s/Remove_Overlapping.png"%(output_folder), imgContours)
            
        # end if # show steps #####################################################################

        
        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        intLenOfLongestListOfChars = 0
        intIndexOfLongestListOfChars = 0

        """
                # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
            # end if
        # end for

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        #longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

        """
        listOfListsOfMatchingCharsInPlate = sorted(listOfListsOfMatchingCharsInPlate,key=lambda x:len(x))
        # All the left plates till now are elligible to be potential part of a number plate
        if len(listOfListsOfMatchingCharsInPlate) > 1:
            longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[-2:]
            longestListOfMatchingCharsInPlate = sorted(longestListOfMatchingCharsInPlate,key=lambda x:x[0].intCenterY)
        else:
            longestListOfMatchingCharsInPlate = [listOfListsOfMatchingCharsInPlate[0]]
        
        if showSteps == True or save_intermediate == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for ListOfCharsInPlate in listOfListsOfMatchingCharsInPlate:
                for matchingChar in ListOfCharsInPlate:
                    contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (255,255,255))

        if showSteps == True:
            cv2.imshow("The_Longest_list_of_matching_chars", imgContours)
            cv2.waitKey(0)
        # end if # show steps #####################################################################

        if save_intermediate == True: # show steps ###################################################
            cv2.imwrite("%s/The_Longest_list_of_matching_chars.png"%(output_folder), imgContours)
            

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate,save_intermediate,output_folder,showSteps)
        if showSteps == True:
            cv2.destroyAllWindows()
        listOfPossiblePlates_refined.append(possiblePlate)

        if showSteps == True: # show steps ###################################################
            print("chars found in plate number " + str(intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            intPlateCounter = intPlateCounter + 1
            cv2.waitKey(0)
        # end if # show steps #####################################################################

    # end of big for loop that takes up most of the function

    if showSteps == True:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    # end if

    return listOfPossiblePlates_refined # we return the list of plates with the probable plate number of each plate.

###################################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    '''
    This fucntion extracts all the contours from the number plate image and groups the relevant contours into a list
    and returns the list
    @input: Grayscale and thresholded image of the numberplate
    @output: List of relevant contours
    '''
    listOfPossibleChars = []                        # this will be the return value
    contours = []
    imgThreshCopy = imgThresh.copy()

            # find all contours in plate
    imgContours, contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    
    for contour in contours:                        # for each contour
        possibleChar = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(possibleChar):              # if contour is a possible char, note this does not 
            listOfPossibleChars.append(possibleChar)       # compare to other chars (yet) add to list of possible chars
        # end if
    # end if

    return listOfPossibleChars
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
    '''
    This fucntion checks if a given countour is relevant for number recognition
    '''
    
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


###################################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):

    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    listOfListsOfMatchingChars = []                  # this will be the return value


    for possibleChar in listOfPossibleChars:                        # for each possible char in the one big list of chars

        # find all chars in the big list that match the current char
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        
        
        listOfMatchingChars.append(possibleChar)                # also add the current char to current possible list of matching chars
        
        # if current possible list of matching chars is not long enough to constitute a possible plate
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:     
            continue
        listOfListsOfMatchingChars.append(listOfMatchingChars)
        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
        
        # recursive call
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      
        
        # for each list of matching chars found by recursive call
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)
        break;



    return listOfListsOfMatchingChars
# end function

###################################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char, and return those matching chars as a list
    listOfMatchingChars = []                # this will be the return value

    for possibleMatchingChar in listOfChars:                # for each char in big list
        if possibleMatchingChar == possibleChar:    # if the char we attempting to find matches for is the exact same char
                                                    # as the char in the big list we are currently checking
                                                    # then we should not include it in the list of matches b/c that would
                                                    # end up double including the current char
            continue                                # so do not add to list of matches and jump back to top of for loop
        # end if
        
        # compute stuff to see if chars are a match
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)

        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)

        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)

        # check if chars match
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):

            # if the chars are a match, add the current char to list of matching chars
            listOfMatchingChars.append(possibleMatchingChar)        
            
        # end if
    # end for

    return listOfMatchingChars                  # return result
# end function


def distanceBetweenChars(firstChar, secondChar):
    '''
    use Pythagorean theorem to calculate distance between two chars
    '''
    intX = abs(firstChar.intCenterX - secondChar.intCenterX)
    intY = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((intX ** 2) + (intY ** 2))


def angleBetweenChars(firstChar, secondChar):
    '''
    use basic trigonometry (SOH CAH TOA) to calculate angle between chars
    '''

    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    # check to make sure we do not divide by zero if the center X positions are equal, 
    # float division by zero will cause a crash in Python
    if fltAdj != 0.0:                           
        fltAngleInRad = math.atan(fltOpp / fltAdj)      # if adjacent is not zero, calculate angle

    else:
        # if adjacent is zero, use this as the angle, this is to be consistent with the C++ version of this program
        fltAngleInRad = 1.5708                          
    # end if

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       # calculate angle in degrees

    return fltAngleInDeg
# end function
###################################################################################################


def removeInnerOverlappingChars(listOfMatchingChars):
    '''
    if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller) char,
    this is to prevent including the same char twice if two contours are found for the same char,
    for example for the letter 'O' both the inner ring and the outer ring may be found as contours, 
    but we should only include the char once    
    '''
  
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:        # if current char and other char are not the same char . . .
                
                # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not already removed on a previous pass, remove it
                    
                    # if current char is smaller than other char
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         

                        # if current char was not already removed on a previous pass . . .
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:        

                            # then remove current char      
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)   
                        # end if
                    else:                                              # else if other char is smaller than current char

                        # if other char was not already removed on a previous pass . . .
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                

                            # then remove other char
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           
                        # end if
                    # end if
                # end if
            # end if
        # end for
    # end for

    return listOfMatchingCharsWithInnerCharRemoved
# end function

###################################################################################################

def recognizeCharsInPlate(imgThresh, ListOflistOfMatchingChars,save_intermediate,output_folder,showSteps):
    '''
    This function performs the actual char recognition
    @input: Thresholded image of the number plate, List of all the one line text contours
    @output: The full length text on the number plate(string).
    '''

    strChars = ""               # this will be the return value, the chars in the lic plate

    height, width = imgThresh.shape
    imgThreshColor = np.zeros((height, width, 3), np.uint8)
    #imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_BGR2HSV)
    #imgHue, imgSaturation, imgThresh = cv2.split(imgHSV)
    #cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #imgThreshColor = imgThresh.copy()
    #imgThreshColor = cv2.resize(imgThreshColor, (0, 0), fx = 1.6, fy = 1.6)

    # Binary inverting the thresholded image
    thresholdValue, imgThresh = cv2.threshold(imgThresh, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #imgThresh = cv2.fastNlMeansDenoising(imgThresh,None,10,10,7,21)

    # Getting the RGB form of thresholded into imgThreshColor
    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)
    
    # Make a copy
    imgThreshColor_plot = imgThreshColor.copy()
    
    # String recognized from each line of the number plate
    for listOfMatchingChars in ListOflistOfMatchingChars:

        # sort chars from left to right
        listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        

        for currentChar in listOfMatchingChars:                                         # for each char in plate
            pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
            pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

            cv2.rectangle(imgThreshColor_plot, pt1, pt2, (255,0,0), 2)           # draw green box around the char
            
            
                
            
            # crop char out of threshold image
            imgROI = imgThreshColor[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
            
            # Add border around the image
            imgROI = cv2.copyMakeBorder(imgROI,8,8,8,8,cv2.BORDER_CONSTANT,value = [255,255,255])

                    # crop char out of threshold image
            
            imgROI = cv2.cvtColor(imgROI, cv2.COLOR_BGR2GRAY)
            imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT),interpolation=cv2.INTER_LINEAR)           # resize image, this is necessary for char recognition
            
            # Pick the number of channels from the currently loaded model.
            channels = model.input.shape[-1].value
            
            img=np.reshape(imgROIResized,[1,64,64,channels])

            classes=model.predict_classes(img)
            
            if classes[0]<10:
                strCurrentChar = chr(classes[0]+48) # get character from results
            else:
                strCurrentChar = chr(classes[0]+55)    # get character from results
            
            strChars = strChars + strCurrentChar                        # append current char to full string

            if showSteps == True:
                cv2.imshow('The Plate',imgThreshColor_plot)
                print(strChars)
                cv2.waitKey(0)
            
            if save_intermediate == True:
                cv2.imwrite('%s/The Plate.png'%(output_folder),imgThreshColor_plot)

        strChars = strChars + ' '

        # end for

    if showSteps == True: # show steps #######################################################
        cv2.imshow("full annotated image", imgThreshColor_plot)
        cv2.waitKey(0)

    if save_intermediate == True:
        cv2.imwrite('%s/full annotated image.png'%(output_folder),imgThreshColor_plot)
    # end if # show steps #########################################################################

    return strChars
# end function


# ## Helper code
def load_image_into_numpy_array(image):
    '''
    This function converts the image into the numpy array for prediction
    '''

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection
def run_inference_for_single_image(image, graph):
    '''
    This function will detect the number plate in the image using the SSD trained model

    @input: Image to process, tensorflow graph
    @output: List containing all the information
    '''

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()

            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            # Do the preprocessing for detection mask
            # The following processing is only for single image
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                
                # Follow the convention by adding back the batch dimension
                
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})


            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


# Image Segmentation
def CapturePlatesFromImage(image):
    # Loading the model
    MODEL_NAME = 'plate_detector' # This is the model we will use here.
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(os.path.abspath(__file__), '..', MODEL_NAME, 'frozen_inference_graph.pb') # Path to save the downloaded model.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    
    # Loading The image
    img_Original = cv2.imread(image)
    img = Image.open(image)    
    image_np = load_image_into_numpy_array(img)
    
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    d = output_dict['detection_boxes'][0].tolist()
    
    (ymin, xmin, ymax, xmax) = d
    
    im_width, im_height = img.size
    
    (left, right, top, bottom) = (xmin*im_width, xmax*im_width,ymin*im_height, ymax*im_height)
    
    imgResult = img_Original[math.floor(top):math.ceil(bottom),math.floor(left):math.ceil(right)]
    
    return imgResult

def main(img_path,save_intermediate=False,output_folder=False,showSteps=False):
    
    # Get the Plates from the Image
    img_plate = CapturePlatesFromImage(img_path)

    # Show the results
    if showSteps == True:
        cv2.imshow('Image',img_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # save the result
    if save_intermediate == True:
        cv2.imwrite('%s/Image.png'%(output_folder),img_plate)

    # Send each plate for the text detection phase
    Plates = []
    possiblePlate = PossiblePlate.PossiblePlate()

    possiblePlate.imgPlate = img_plate
    
    Plates.append(possiblePlate)
    
    Refined_plate = detectCharsInPlates(Plates,save_intermediate,output_folder,showSteps)
    
    if len(Refined_plate) ==0:
    	print('No Plate Found')
    	return ' ',img_plate
    else:
        return Refined_plate[0].strChars,img_plate

if __name__ == '__main__':
    
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ImagePath", required=True,help="Path to where the image to be processed is placed")
    ap.add_argument("-s","--Save_intermediate",required=False,default=False,type=bool,help="Saved the intermediate results into a folder")
    ap.add_argument("-o","--output_folder",required=False,default=None,help="Folder where the intermediate images will be saved")
    ap.add_argument("-sh","--ShowSteps",required=False,default=False,type=bool,help="To see the intermediate results")

    args = ap.parse_args()

    if args.Save_intermediate and args.output_folder == None:
        print("Output folder to save the intemedaite images should be specified")
        exit()
    
    if args.output_folder!=None and os.path.exists(args.output_folder) == False:
        os.mkdir(args.output_folder)

    number_plate_text,plate = main(args.ImagePath,args.Save_intermediate,args.output_folder,args.ShowSteps)
    print(number_plate_text)
    
