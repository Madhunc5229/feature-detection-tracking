import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def readImages(folder, num_images):
  
  arr_images = []
  for i in range(num_images):
    arr_images.append(cv2.imread(f'data/LK_images/hotel.seq{i}.png'))
  return np.array(arr_images, dtype=np.float32)


def imgfilter(image, filter):
    # YOUR CODE HERE
    img = cv2.filter2D(image, -1, filter)
    return img

def toUint8(image):
    img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)  # normalize the data to 0 - 1
    return img

def imgGradients(img, filter_size, sigma):
  soble_x = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float64)
  soble_y = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]],dtype=np.float64)
  
  #parameters for gaussian blurring the image
  filter_size = filter_size
  sigma = sigma

  #convert to gray image
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  img = cv2.GaussianBlur(img,filter_size,sigma)
  Ix = imgfilter(img,soble_x)
  Iy = imgfilter(img,soble_y)
  Ixx = np.square(Ix)
  Iyy = np.square(Iy)
  Ixy = np.multiply(Ix,Iy)

  Ixx = cv2.GaussianBlur(Ixx,filter_size,sigma)
  Iyy = cv2.GaussianBlur(Iyy,filter_size, sigma)
  Ixy = cv2.GaussianBlur(Ixy,filter_size, sigma)
  
  return Ix,Iy, Ixx, Iyy, Ixy

def getKeypoints(img, tau, filter_size, sigma):
  '''
  Detecting keypoints using Harris corner criterion
  img: input image
  tau: threshold 
  
  output: (N,2) array of [x,y] keypoints
  '''
  img_clr = np.copy(img)
  filter_size = filter_size
  sigma = sigma
  Ix, Iy, Ixx, Iyy, Ixy = imgGradients(img,filter_size,sigma)

  #converting img to uint8 format
  img_clr = toUint8(img_clr)

  # determinant
  det_H = Ixx * Iyy - Ixy ** 2
  # trace
  trace_H = Ixx + Iyy
  

  harris_score = det_H/trace_H
  harris_score = cv2.normalize(harris_score, None, 0, 1, cv2.NORM_MINMAX)

  num_of_points = 0
  win_size = 7
  offset = win_size//2

  #Non Maximum Suppression
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      try:
        window = harris_score[i-offset:i+offset,j-offset:j+offset]
        #suppress all the values less than maximum in the window
        window[window<np.max(window)] = 0
        harris_score[i-offset:i+offset,j-offset:j+offset] = window
      except:
        IndexError

  #vizualize the corners
  key_points = []
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if harris_score[i,j] > tau:
        key_points.append(np.array([i,j],dtype=np.float64))
        img_clr = cv2.circle(img_clr,(j,i),1, (0,0,255),2)
        num_of_points +=1

  #convert key points to array of Nx2
  key_points = np.array(key_points,dtype=np.float64)

  print("No. of key points: ", num_of_points)
  plt.figure(figsize=(6,8))
  plt.imshow(img_clr)
  cv2.imwrite('/content/corners.png',img_clr)
  
  # YOUR CODE HERE
  return key_points, img_clr


def getNextPoints(x, y, im1, im2, ws):
  '''
  Iterative Lucas-Kanade feature tracking
  x,  y : initialized keypoint position in im2
  ws: patch window size

  output: tracked keypoint positions in im2
  '''
  num_points = x.shape[0]
  x_new , y_new = x ,y 

  filter_size = (3,3)
  sigma = 2
  
  u = []
  v = []

  #get gradients of image 1
  Ix, Iy, Ixx, Iyy, Ixy = imgGradients(im1,filter_size,sigma)
  
  im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

  im1 = cv2.GaussianBlur(im1,filter_size,sigma)
  im2 = cv2.GaussianBlur(im2,filter_size,sigma)
  It = im2-im1

  out_of_frame = []
  iterations = 8
  for iter in range(iterations):
    for i in range(num_points):
      if 0 < x_new[i] < im1.shape[0] and 0 < y_new[i] < im1.shape[1]:

        ix_w = (cv2.getRectSubPix(Ix,(ws,ws),(x[i],y[i]))).flatten()
        iy_w = (cv2.getRectSubPix(Iy,(ws,ws),(x[i],y[i]))).flatten()
        it_w = (cv2.getRectSubPix(It,(ws,ws),(x[i],y[i]))).flatten()

        ixx_w = np.sum(np.square(ix_w))
        iyy_w = np.sum(np.square(iy_w))
        ixy_w = np.sum(ix_w*iy_w)
        ixt_w = np.sum(ix_w*it_w)
        iyt_w = np.sum(iy_w*it_w)

        A = np.array([[ixx_w, ixy_w],[ixy_w, iyy_w]])
        b = -np.array([[ixt_w],[iyt_w]])
        
        d = np.matmul(np.linalg.pinv(A), b) 

        x_new[i] += d[0][0]
        y_new[i] += d[1][0]
      else:
        out_of_frame.append(np.array([x_new[i],y_new[i]]))

    
  return x_new,y_new, out_of_frame


def trackPoints(pt_x, pt_y, im, ws):
  num_kp = pt_x.shape[0]
  no_imgs = im.shape[0]
  track_x = np.zeros((num_kp,no_imgs))
  track_y = np.zeros((num_kp,no_imgs))
  out_of_frame_pts = []

  track_x[:,0] = pt_x[:]
  track_y[:,0] = pt_y[:]

  for t in range(no_imgs-1):
    track_x[:,t+1], track_y[:,t+1], out_of_frame = getNextPoints(x=track_x[:,t],y=track_y[:,t],im1=im[t],im2=im[t+1],ws=ws)
    for i in range(len(out_of_frame)):
      out_of_frame_pts.append(out_of_frame[i])
  return track_x, track_y, out_of_frame_pts

def getEdgePoints(im1,im2):
  im1_pts = np.nonzero(im1)
  im2_pts = np.nonzero(im2)
  return im1_pts, im2_pts

def getInitialTfm(x1,y1,x2,y2):

  x1_mean,y1_mean = np.mean(x1), np.mean(y1)
  x2_mean,y2_mean = np.mean(x2), np.mean(y2)

  x1_var, y1_var = np.var(x1), np.var(y1)
  x2_var, y2_var = np.var(x2), np.var(y2)

  scale_x = x1_var/x2_var
  scale_y = y1_var/y2_var


  trans_x = x1_mean - x2_mean
  trans_y = y1_mean - y2_mean

  return trans_x, trans_y, scale_x, scale_y
  
def getMatchingPts(x1,y1,x2,y2):

  match_pts = np.empty((x1.shape[0],2))

  for i in range(x1.shape[0]):
    dist = np.inf
    for j in range(x2.shape[0]):
      if 0 < i < x1.shape[0] and 0 < j < x2.shape[0]: 
        temp_dist = np.sqrt((x1[i]-x2[j])**2 + (y1[i]-y2[j])**2)
        if temp_dist < dist:
          dist = temp_dist
          match_pts[i] = np.array([x2[j], y2[j]])

  return match_pts

def constructA(x,y):
  A = np.empty((2*x.shape[0],6))

  for i in range(A.shape[0]):
      j = i//2
      if i%2 == 0:
        A[i,0] = x[j]
        A[i,1] = y[j]
        A[i,2] , A[i,3], A[i,5]= 0,0,0
        A[i,4] = 1
      else:
        A[i,2] = x[j]
        A[i,3] = y[j]
        A[i,0] , A[i,1], A[i,4]= 0,0,0
        A[i,5] = 1
  return A

def constructb(pts):
  b = np.empty((2*pts.shape[0],1))
  for i in range(b.shape[0]):
    j = i//2
    if i%2 == 0:
      b[i] = pts[j,0]
    else:
      b[i] = pts[j,1]
  return b

def evalAlignment(aligned1, im2):
  '''
  Computes the error of the aligned image (aligned1) and im2, as the
  average of the average minimum distance of a point in aligned1 to a point in im2
  and the average minimum distance of a point in im2 to aligned1.
  '''
  d2 = ndimage.distance_transform_edt(1-im2) #distance transform
  err1 = np.mean(np.mean(d2[aligned1 > 0]))
  d1 = ndimage.distance_transform_edt(1-aligned1)
  err2 = np.mean(np.mean(d2[im2 > 0]))
  err = (err1+err2)/2
  return err

def displayAlignment(im1, im2, aligned1, thick=False):
  '''
  Displays the alignment of im1 to im2
     im1: first input image to alignment algorithm (im1(y, x)=1 if (y, x) 
      is an original point in the first image)
     im2: second input image to alignment algorithm
     aligned1: new1(y, x) = 1 iff (y, x) is a rounded transformed point from the first time 
     thick: true if a line should be thickened for display
  ''' 
  if thick:
    # for thick lines (looks better for final display)
    dispim = np.concatenate((cv2.dilate(im1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(aligned1.astype('uint8'), np.ones((3,3), np.uint8), iterations=1), \
                             cv2.dilate(im2.astype('uint8'), np.ones((3,3), np.uint8), iterations=1)), axis=-1)
  else:
    # for thin lines (faster)
    dispim = np.concatenate((im1, aligned1, im2), axis = -1)
  return dispim
  


def align_shape(im1, im2):
  '''
  im1: input edge image 1
  im2: input edge image 2

  Output: transformation T [3] x [3]
  '''
  im1_pts, im2_pts = getEdgePoints(im1,im2)

  x1 = im1_pts[0]
  y1 = im1_pts[1]
  #img1 non zero points 2 X N
  im1_xy = np.vstack((x1,y1))
  x2 = im2_pts[0]
  y2 = im2_pts[1]
  #img2 non zero points 2 X N
  im2_xy = np.vstack((x2,y2))
  iterations = 25

  tx, ty, sx, sy = getInitialTfm(x1,y1,x2,y2)
  m1 = 1
  m2 = 0
  m3 = 0
  m4 = 1
  tfm_mat = np.array([[m1, m2,tx],[m3, m4, ty]])

  #initial transformation
  #i = 0 to N1
  for i in range(im1_xy.shape[1]):
    try:
      x1[i] = m1*x1[i] + m2*y1[i] + tx 
      y1[i] = m3*x1[i] + m4*y1[i] + ty 
    except:
      ValueError
      OverflowError

  for iter in range(iterations):
  #match_pts is Nx 2 array containing matching points in im2
    match_pts1 = getMatchingPts(x1,y1,x2,y2)

    A_mat = constructA(x1,y1)

    B = constructb(match_pts1)
    X = np.matmul(np.linalg.pinv(A_mat), B)

    m1,m2,m3,m4,tx,ty = X[0],X[1],X[2],X[3],X[4],X[5]
    try:
      for i in range(x1.shape[0]):
        x1[i] = m1*x1[i] + m2*y1[i] + tx 
        y1[i] = m3*x1[i] + m4*y1[i] + ty 
    except:
      ValueError
      OverflowError

  return x1,y1
  
def calculateSSD(img1, img2):

  if img1.shape != img2.shape:
    return -1
  else:
    X = np.subtract(img1,img2)
    ssd = np.sum(np.square(X))
    return ssd

