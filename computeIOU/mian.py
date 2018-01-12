#-*-coding:utf-8-*-
#-*- coding:utf-8 -*-
from PIL import Image
import pylab
import os
import cv2
import numpy as np
def computeIou(resultImage,resultColour,groundTruth,truthColour):
    height=resultImage.shape[0]
    width=resultImage.shape[1]
    #compute
    resultAll=0
    truthAll=0
    correct=0
    for i in range(height):
        for j in range(width):
            if resultImage[i][j][2]==resultColour:
                resultAll=resultAll+1
            if groundTruth[i][j][2]==truthColour:
                truthAll=truthAll+1
            if groundTruth[i][j][2] == truthColour and resultImage[i][j][2]==resultColour:
                correct = correct + 1

    correctness=float(correct)/float(resultAll)
    completeness = float(correct) / float(truthAll)
    return [correctness,completeness]

label='/media/yqyi/33E48B2644A4DCB7/myPaper/windowsEx/result/cliped2labelSH.png'
labelImage=cv2.imread(label)
result='/media/yqyi/33E48B2644A4DCB7/myPaper/windowsEx/result/enviResult.tif'
resultImage=cv2.imread(result)
ori='/media/yqyi/33E48B2644A4DCB7/myPaper/windowsEx/result/cliped2testPython.jpg'
oriImage=cv2.imread(ori)

print(computeIou(resultImage,255,labelImage,255))


emptyImage = oriImage

height = oriImage.shape[0]
width = oriImage.shape[1]
for i in range(height):
    for j in range(width):
        if resultImage[i][j][2] == 255:
            emptyImage[i][j]=[177,156,242]
cv2.imwrite('/media/yqyi/33E48B2644A4DCB7/myPaper/windowsEx/result/final.jpg',emptyImage)

