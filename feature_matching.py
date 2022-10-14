import numpy as np
from common import *
import cv2
from scipy.io import loadmat

img1 = cv2.imread('data/stop1.jpg')
img2 = cv2.imread('data/stop2.jpg')

# inside the sift are:
# Descriptor1(128x597), Descriptor2(128, 2397): SIFT features from image 1 and image 2
# Frame1(4, 597), Frame2(4, 2397): position, scale, rotation of keypoints
data = loadmat('data/SIFT_features.mat')
Frame1 = data['Frame1']
Descriptor1 = data['Descriptor1']
Frame2 = data['Frame2']
Descriptor2 = data['Descriptor2']

d_thres = 0.7
# matches: a 2 x N array of indices that indicates which keypoints from image
# 1 match which points in image 2
ratio_matches = []
dist_matches = []

nb_1 = 0
bad_matches = 0
good_matches = 0

for i in range(Descriptor1.shape[1]):
  d1 = np.inf 
  d2 = np.inf 
  for j in range(Descriptor2.shape[1]):
    dist = calculateSSD(Descriptor2[:,j],Descriptor1[:,i])
    if dist < d1:
      d1 = dist
      nb_1 = j
    if d1 < dist < d2:
      d2 = dist
    if dist < 3500:
      dist_matches.append(np.array([i,j]))
  if d1/d2 < d_thres:
    good_matches += 1
    ratio_matches.append(np.array([i,nb_1]))
  else:
    bad_matches +=1

ratio_matches = np.array(ratio_matches)
dist_matches = np.array(dist_matches)

img1_pts_ratio = np.empty((ratio_matches.shape[0],2))
img2_pts_ratio = np.empty((ratio_matches.shape[0],2))

img1_pts_dist = np.empty((dist_matches.shape[0],2))
img2_pts_dist = np.empty((dist_matches.shape[0],2))

for i in range(ratio_matches.shape[0]):
  img1_pts_ratio[i,0] = Frame1[0,ratio_matches[i,0]]
  img1_pts_ratio[i,1] = Frame1[1,ratio_matches[i,0]]
  img2_pts_ratio[i,0] = Frame2[0,ratio_matches[i,1]]
  img2_pts_ratio[i,1] = Frame2[1,ratio_matches[i,1]]

for i in range(dist_matches.shape[0]):
  img1_pts_dist[i,0] = Frame1[0,dist_matches[i,0]]
  img1_pts_dist[i,1] = Frame1[1,dist_matches[i,0]]
  img2_pts_dist[i,0] = Frame2[0,dist_matches[i,1]]
  img2_pts_dist[i,1] = Frame2[1,dist_matches[i,1]]

# YOUR CODE HERE
padding = np.zeros((203,355,3),dtype=np.uint8)
img1 = np.vstack((img1,padding))
ratio_img = np.hstack((img1,img2))
dist_img = np.copy(ratio_img)

for i in range(ratio_matches.shape[0]):
  ratio_img = cv2.circle(ratio_img,(int(round(img1_pts_ratio[i,0],0)),int(round(img1_pts_ratio[i,1],0))),3, (0,255,0),-1) 
  ratio_img = cv2.circle(ratio_img,(int(round(img2_pts_ratio[i,0]+355,0)),int(round(img2_pts_ratio[i,1],0))),3, (0,255,0),-1) 
  cv2.line(ratio_img, (int(round(img1_pts_ratio[i,0],0)),int(round(img1_pts_ratio[i,1],0))), (int(round(img2_pts_ratio[i,0]+355,0)),int(round(img2_pts_ratio[i,1],0))), (0,255,0), 2) 

for i in range(dist_matches.shape[0]):
  dist_img = cv2.circle(dist_img,(int(round(img1_pts_dist[i,0],0)),int(round(img1_pts_dist[i,1],0))),3, (0,255,0),-1) 
  dist_img = cv2.circle(dist_img,(int(round(img2_pts_dist[i,0]+355,0)),int(round(img2_pts_dist[i,1],0))),3, (0,255,0),-1) 
  cv2.line(dist_img, (int(round(img1_pts_dist[i,0],0)),int(round(img1_pts_dist[i,1],0))), (int(round(img2_pts_dist[i,0]+355,0)),int(round(img2_pts_dist[i,1],0))), (0,255,0), 2) 


cv2.imshow('Matches using distance ratio',ratio_img)
cv2.imwrite('results/Matches_using_distance_ratio.png',ratio_img)
cv2.waitKey(0)

cv2.imshow('Matches using neighboring distance',dist_img)
cv2.imwrite('results/Matches_using_neighboring_distance.png',dist_img)
cv2.waitKey(0)