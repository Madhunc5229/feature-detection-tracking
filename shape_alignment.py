import numpy as np
from common import *
import cv2
import time

imgPath = 'data/shape_alignment/'

objList = ['apple', 'bat', 'bell', 'bird', 'Bone', 'bottle', 'brick', \
    'butterfly', 'camel', 'car', 'carriage', 'cattle', 'cellular_phone', \
    'chicken', 'children', 'device7', 'dog', 'elephant', 'face', 'fork', 'hammer', \
    'Heart', 'horse', 'jar', 'turtle']

numObj = len(objList)

im1_path = imgPath + objList[2] + '_1.png'
im2_path = imgPath + objList[2] + '_2.png'
im1 = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)

im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

output = np.zeros_like(im1)
stat_time = time.time()
x1, y1 = align_shape(im1,im2)
end_time = time.time()
output[x1,y1] = 255 

error = evalAlignment(output,im2)
print("error for: ",objList[2]," ", error)
print("Run time for: ",objList[2]," ", end_time-stat_time)
compareImg = displayAlignment(im1, im2, output, thick=True)

cv2.imshow('Image1 aligned Image Image 2',compareImg)
cv2.waitKey(0)
