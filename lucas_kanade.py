import numpy as np
from common import *
import cv2

# read all 51 sequences of images
folder = '/data/LK_images'
im = readImages(folder, 51)

filter_size = (5,5)
sigma = 2
tau = 0.06
key_pts, img_corners = getKeypoints(im[0], tau, filter_size,sigma)

track_x, track_y, out_pts = trackPoints(pt_x=key_pts[:,0], pt_y=key_pts[:,1], im=im, ws=7)

img_tracked = np.copy(toUint8(im[0]))
img_20pts = np.copy(toUint8(im[0]))
img_out_pts = np.copy(toUint8(im[0]))

#display the first set of key points
cv2.imshow('First set of Key Points',img_corners)
cv2.waitKey(0)

#display second set of key points
for i in range(track_x.shape[0]):
  cv2.circle(img_corners,(int(round(track_y[i,1],0)),int(round(track_x[i,1],0))),1,(0,0,255),2)

cv2.imshow('Second set of Tracked Points',img_corners)
cv2.imwrite('results/Second_set_of_Tracked_Points.png',img_corners)
cv2.waitKey(0)


#display sequence for 20 points
for i in range(20):
  for j in range(track_x.shape[0]):
    cv2.circle(img_20pts,(int(round(track_y[j,i],0)),int(round(track_x[j,i],0))),1,(0,0,255),-1)

cv2.imshow('Progress for 20 points',img_20pts)
cv2.imwrite('results/Progress_for_20_points.png',img_20pts)
cv2.waitKey(0)

#display full sequence image points
for i in range(51):
  for j in range(track_x.shape[0]):
    cv2.circle(img_tracked,(int(round(track_y[j,i],0)),int(round(track_x[j,i],0))),1,(0,0,255),-1)

cv2.imshow('Full sequence',img_tracked)
cv2.imwrite('results/Full_sequence.png',img_tracked)
cv2.waitKey(0)

