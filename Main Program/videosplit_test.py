# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import os
import time
import shutil

def Launch(image):
    start = time.time()
    cap = cv2.VideoCapture(image)

    fps = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( "total frames = : ",totalFrames )
    videolength = totalFrames/fps
    
    try:
        if os.path.exists('data'):
            shutil.rmtree('data', ignore_errors=True)
        os.makedirs('data')

    except OSError:
        print('Error: Creating data')
       
    count = 0
    success = True
    framesWeNeed = 30
    interval = round(totalFrames/framesWeNeed)
    while (success and count < 1):
        
        #uncomment this to bottleneck number of frames to 60.
        i = 0
        '''
        while(i<interval-1):
            a,b = cap.read()
            i += 1
		 '''
        success, frame = cap.read()
        shape = frame.shape[:2]
        print (shape)
        '''
        frame = cv2.resize(frame,(shape[0],shape[1]),interpolation = cv2.INTER_LINEAR)
        shape_new = frame.shape[:2]
        
        print ('shape_new: ',shape_new)
        '''
        '''
        #rotate 90 clockwise
        frame=cv2.transpose(frame)
        frame=cv2.flip(frame,flipCode=1)
        '''
        #cv2.imshow('frame',frame)
        #cv2.waitKey(0)
        
        name = './data/frame' + str(count) + '.jpg'
        # print ('Creating ' + name)
        cv2.imshow('img',frame)
        cv2.waitKey(0)
        cv2.imwrite(name, frame)
        
        count += 1
    '''
    # to remove the last undesired image
    name = './data/frame' + str(count - 1) + '.jpg'
    os.remove(name)
    
    print('\ntime taken = ' + str(time.time() - start))
    '''
    cap.release()
    cv2.destroyAllWindows()
    
    return ("{0:.2f}".format(videolength),str(totalFrames))
    
Launch('./videos/car13.mp4')
    