from common import *
import cv2

# read all 51 sequences of images
folder = '/data/LK_images'
im = readImages(folder, 51)


filter_size = (5,5)
sigma = 2
tau = 0.06
key_pts, img_corners = getKeypoints(im[0], tau, filter_size,sigma)

cv2.imshow('Corner detectiong',img_corners)
cv2.waitKey(0)