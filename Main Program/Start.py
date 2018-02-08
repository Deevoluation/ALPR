import os
import time

import datetime
import videosplit
import Main
import cv2
import pymongo




def mongo_connection():
    con = pymongo.MongoClient(host)
    col = con[database][collection]
    return col


if __name__ == '__main__':
    name = str(input('Enter the name of the video: '))
    (vdolength,totalFrames) = videosplit.Launch(name)
    os.chdir('data')

    result = {}
    result_imag = {}
    #startTime = datetime.now()
    startTime = time.time()
    for f in os.listdir():
        pred, img = Main.main(f)
        if pred in result.keys():
            result[pred] = result[pred] + 1
        elif pred != ' ':
            result[pred] = 1
            result_imag[pred] = img

    #endTime = datetime.now()
    endTime = time.time()
    executionTime = "{0:.2f}".format(endTime - startTime)
	print('The name plate is :', plate, ' frequency is: ', result[plate])
    cv2.imshow('The plate', img)
    host = 'localhost'
    database = 'ALPR'
    collection = 'videosTest'
    col = mongo_connection()
    
    l = {x: y for y, x in result.items()}
    r = list(sorted(l.keys()))
    index = r[len(r) - 1]
    plate = l[index]
    img = result_imag[plate]
    print('execution time is : ' + executionTime)
    dict = {}
    dict['date and time'] = time.ctime()
    dict['video'] = name
    dict['video length'] = vdolength
    dict['image'] = plate
    dict['Total Frames in video'] = totalFrames
    dict['execution_time'] = executionTime
    dict['frequency ratio'] = "{0:.2f}".format(result[plate] / len(result))
    
    col.insert(dict)
    
    os.chdir('..')
    licensePlatePath = './licensePlates/'+name.split('.')[0]+'.jpg'
    cv2.imwrite(licensePlatePath,img)
    cv2.waitKey(0)
    for i in result.keys():
        print(i, ' : ', result[i])
