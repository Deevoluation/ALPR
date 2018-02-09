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

def Launch(video):
    start = time.time()
    cap = cv2.VideoCapture(video)

    fps = cap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print( "total frames = : ",totalFrames )
    videolength = totalFrames/fps
    
    try:
        if os.path.exists('data'):
            shutil.rmtree('data', ignore_errors=True)
        os.makedirs('data')

    except OSError:
        print('Error: Creating data')

    count = 0
    success = True
    framesWeNeed = 10
    interval = round(totalFrames/framesWeNeed)
    while (success):
        
        #uncomment this to bottleneck number of frames to 30.
        i = 0
        while(i<interval-1):
            a,b = cap.read()
            i += 1
		 
        success, frame = cap.read()
		#comment two lines below to remove rotation clockwise.
        frame=cv2.transpose(frame)
        frame=cv2.flip(frame,flipCode=1)
        #cv2.imshow('frame',frame)
        #cv2.waitKey(0)
        name = './data/frame' + str(count) + '.jpg'
        # print ('Creating ' + name)
        cv2.imwrite(name, frame)
        count += 1
    print( "total frames = : ",count )   
    name = './data/frame' + str(count - 1) + '.jpg'
    os.remove(name)

    print('\ntime taken = ' + str(time.time() - start))

    cap.release()
    cv2.destroyAllWindows()
    
    return ("{0:.2f}".format(videolength),str(totalFrames))
    
#Launch('car1.mp4')
