# -*- coding: utf-8 -*-
#pyplot belongs to matplotlib
import pylab as pl
from roipoly import roipoly
import numpy as np
import os
import cv2
# create image
def clipImage(imagePath,judgerMatrix):
    imageList=os.listdir(imagePath)
    ny, nx = np.shape(judger)

    for imageName in imageList:
        sourceImg = cv2.imread(imagePath+'/'+imageName)
        result1 = sourceImg.copy()
        result2 = sourceImg.copy()
        for i in range(sourceImg.shape[0]):
            for j in range(sourceImg.shape[1]):
                if judgerMatrix[i, j] == False:
                    result1[i, j][0]= 255
                    result1[i, j][1] = 255
                    result1[i, j][2] = 255
                else:
                    result2[i, j][0] = 255
                    result2[i, j][1] = 255
                    result2[i, j][2] = 255
        cv2.imwrite(imagePath+"/cliped1_"+imageName[0:-4]+'.jpg',result1)
        cv2.imwrite(imagePath +"/cliped2_" + imageName[0:-4] + '.jpg',result2)

imagePath="test"
baseImageName="1.jpg"
baseImage = pl.imread(imagePath+'/'+baseImageName)
# show the image
pl.imshow(baseImage, interpolation='nearest', cmap="Greys")
pl.colorbar()
pl.title("left click: line segment         right click: close region")

# let user draw first ROI
ROI1 = roipoly(roicolor='r') #let user draw first ROI

# show the image with the first ROI
pl.imshow(baseImage, interpolation='nearest', cmap="Greys")
pl.colorbar()
ROI1.displayROI()
print("start get mask")
judger=ROI1.getMask(baseImage)
print("start clip")
clipImage(imagePath,judger)
