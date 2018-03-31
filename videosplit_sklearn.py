# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import skvideo.io
import numpy as np
import os
import time
import shutil
import scipy.misc

def Launch(video):
    start = time.time()
    
    videodata = skvideo.io.vreader(video)
    metadata = skvideo.io.ffprobe(video)
    metadata = metadata['video']
      
    size = (metadata['@width'],metadata['@height'])
    fps = metadata['@avg_frame_rate']
    totalFrames = int(metadata['@nb_frames']) 
    videolength = metadata['@duration']
    #print( "total frames = : ",totalFrames )
    
    try:
        if os.path.exists('data'):
            shutil.rmtree('data', ignore_errors=True)
        os.makedirs('data')

    except OSError:
        print('Error: Creating data')

  
    framesWeNeed = 5
    interval = round(totalFrames/framesWeNeed)
    count = 0
    frame_no = 0
    for frame in videodata:
        
        if(count % interval == 0):
            frame_no += 1
            name = './data/frame' + str(frame_no) + '.jpg'
            scipy.misc.imsave(name,frame) 
        count += 1
    
    print( "total frames generated = : ",frame_no)   
    
    print('\ntime taken = ' + str(time.time() - start))
    
    return ("{0:.2f}".format(float(videolength)),str(totalFrames))
    
#Launch('car18.mp4')

