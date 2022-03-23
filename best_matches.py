# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:01:34 2019
@author: W10
"""


import numpy as np
from matplotlib import pyplot as plt
import cv2

METHOD = 'SIFT' # 'SIFT','SURF', 'ORB' 

# Read the images
img1 = cv2.imread('input/img1.ppm')
img2 = cv2.imread('input/img2.ppm')
#cv2.imshow('Input',img1)

# Convert the images to grayscale
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


if METHOD == 'SIFT':
    print('Calculating SIFT features...')
    
    # Create a SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    # Get the keypoints and the descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray,None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray,None)
    # keypoints object includes position, size, angle, etc.
    # descriptors is an array. For sift, each row is a 128-length feature vector

elif METHOD == 'SURF':
    print('Calculating SURF features...')
    surf = cv2.xfeatures2d.SURF_create(4000)
    keypoints1, descriptors1 = surf.detectAndCompute(img1_gray,None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2_gray,None)

elif METHOD == 'ORB':
    print('Calculating ORB features...')
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray,None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray,None)     
    # Note: Try cv2.NORM_HAMMING for this feature
    

# Draw the keypoints
img1 = cv2.drawKeypoints(image=img1, outImage=img1, keypoints=keypoints1, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (0, 0, 255))

img2 = cv2.drawKeypoints(image=img2, outImage=img2, keypoints=keypoints2, 
                        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, 
                        color = (0, 0, 255))


# Display the images
cv2.imwrite('Keypoints1.png', img1)
cv2.imwrite('Keypoints2.png', img2)


# Create a brute-force descriptor matcher object
if METHOD == 'SIFT' or METHOD == 'SURF':
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True) 
elif METHOD == 'ORB' :
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 

# Match keypoints
matches1to2 = bf.match(descriptors1,descriptors2)
# matches1to2 is a DMatch object
# LOOK AT OPENCV DOCUMENTATION AND 
#   LEARN ABOUT THE DMatch OBJECT AND ITS FIELDS, SPECIFICALLY THE STRENGTH OF MATCH
#   matches1to2[0].distance

repetabilityRates=len(matches1to2)/((len(keypoints1)+len(keypoints2))/2)

# Sort according to distance and display the first 40 matches
matches1to2 = sorted(matches1to2, key = lambda x:x.distance)
matchPercent=0.01
numberOfMatches = int(len(matches1to2) * matchPercent)
bestMatches = matches1to2[:numberOfMatches]

#shows first 40 matches
imMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2[:40], None)
cv2.imwrite("40Matches.png", imMatches)


#shows good matches best %1 of the all matches
img4 = cv2.drawMatches(img1,keypoints1,img2,keypoints2,bestMatches,img2,flags=2)
cv2.imwrite('BestMatches.png',img4)

# Extract location of matches
pointsOne = np.zeros((len(matches1to2), 2), dtype=np.float32)
pointsTwo = np.zeros((len(matches1to2), 2), dtype=np.float32)
 
for i, match in enumerate(matches1to2):
    pointsOne[i, :] = keypoints1[match.queryIdx].pt
    pointsTwo[i, :] = keypoints2[match.trainIdx].pt

#finding the homography matrix
h, mask = cv2.findHomography(pointsOne, pointsTwo, cv2.RANSAC)

# Use homography to warp img1 onto img2 by using homography matrix above.
height, width = img2_gray.shape
im1Reg = cv2.warpPerspective(img1_gray, h, (width, height))
cv2.imwrite('warpedImage.png',im1Reg)

difimg=cv2.subtract(im1Reg,img2_gray)
cv2.imwrite('subtractedImage.png',difimg)







