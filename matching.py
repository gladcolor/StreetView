import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img1 = cv.imread(r"G:\My Drive\right.jpg",cv.IMREAD_GRAYSCALE)          # queryImage
img2 = cv.imread(r"G:\My Drive\left.jpg",cv.IMREAD_GRAYSCALE) # trainImage
# Initiate ORB detector


# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
surf = cv.xfeatures2d.SURF_create(100)
orb =  cv.ORB_create()

kp_orb1 = orb.detect(img1,None)
kp_orb2 = orb.detect(img2,None)
kp_orb1, orb_des1 = orb.compute(img1, kp_orb1)
kp_orb2, orb_des2 = orb.compute(img2, kp_orb2)
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

surf_kp1, surf_des1 = surf.detectAndCompute(img1, None)
surf_kp2, surf_des2 = surf.detectAndCompute(img2, None)
surf.setHessianThreshold(5)
surf.setUpright(True)
surf.getExtended()
print(len(kp1))
print(len(kp2))

print(len(surf_kp1))
print(len(surf_kp2))
print( surf.getHessianThreshold() )
print( surf.getUpright() )
# Find size of descriptor
print( surf.descriptorSize() )
surf.getExtended()
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

matches_surf = bf.knnMatch(surf_des1, surf_des2, k=2)

matches_orb = bf.knnMatch(orb_des1, orb_des2, k=2)

print( surf.descriptorSize() )
print( surf_des2.shape )

# Apply ratio test
good = []
for m,n in matches_surf:
    if m.distance < 0.65*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,surf_kp1,img2,surf_kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(30, 20))
plt.imshow(img3)
plt.show()


