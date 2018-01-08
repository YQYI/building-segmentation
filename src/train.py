import os
import cv2
import shutil
import numpy as np
import time
def clipImage(imageClipedPath,savePath,side,backGroundValue):
    image=cv2.imread(imageClipedPath)
    suffix=imageClipedPath[-4:]
    padH=int(side-(image.shape[0]/side))
    padW=int(side-(image.shape[1]/side))
    imagePaded=cv2.copyMakeBorder(image, 0, padH, 0, padW, cv2.BORDER_CONSTANT, value=backGroundValue)
    splitH=int(imagePaded.shape[0]/side)
    splitW=int(imagePaded.shape[1]/side)
    for i in range(splitH):
        for j in range(splitW):
            piece=imagePaded[side*i:side*i+side,side*j:side*j+side]
            cv2.imwrite(savePath+'/'+str(i)+'_'+str(j)+suffix,piece)

def ifNeedDelete(labelClipedPath,imageClipedPath,labelList,backGroundValue,num):
    for labelName in labelList:
        print("process "+str(num)+" is checking " + labelName)
        labelImage = cv2.imread(labelClipedPath + '/' + labelName)
        height = labelImage.shape[0]
        width = labelImage.shape[1]
        noDelete = False
        for i in range(height):
            for j in range(width):
                if labelImage[i][j][0] != backGroundValue[0]:
                    noDelete = True
                    break
            if noDelete == True:
                break
        if noDelete == False:
            os.remove(labelClipedPath + '/' + labelName)
            os.remove(imageClipedPath + '/' + labelName[0:-4] + '.jpg')

def transformDimension(labelClipedPath,colourMap,labelCliped1DPath):
    labelClipedNameList=os.listdir(labelClipedPath)
    for aName in labelClipedNameList:
        image=cv2.imread(labelClipedPath+'/'+aName)
        image1D = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
        for c, i in colourMap.items():
            m = np.all(image == np.array(c).reshape(1, 1, 3), axis=2)
            image1D[m] =i
        cv2.imwrite(labelCliped1DPath+'/'+aName,image1D)

##########所有的路径


####先切割训练样本
print("start clip")
backGroundValue=(255,255,255)
imageSource='../train/image/trainImage.jpg'
labelSource='../train/label/trainLabel.png'
side=321
clipPath ='../train/clipResult'
if os.path.exists(clipPath):
    shutil.rmtree(clipPath)
imageClipedPath=clipPath+'/image'
labelClipedPath=clipPath+'/label'
os.makedirs(imageClipedPath)
os.makedirs(labelClipedPath)
print("clip train image")
clipImage(imageSource,imageClipedPath,side,backGroundValue)
print("clip label image")
clipImage(labelSource,labelClipedPath,side,backGroundValue)

####剔除无用样本
print("start check!")
labelNameList=os.listdir(labelClipedPath)
##获取cpu数量
from multiprocessing import cpu_count
cpuNum=cpu_count()
print("共有"+str(cpuNum)+"个cpu")
listPieceNum=int(len(labelNameList)/cpuNum)
import multiprocessing
processList=[]
for i in range(cpuNum):
    if i == cpuNum-1:
        aProcess=multiprocessing.Process(target=ifNeedDelete,args=(labelClipedPath,imageClipedPath,labelNameList[listPieceNum*i:],backGroundValue,i))
    else:
        aProcess = multiprocessing.Process(target=ifNeedDelete,args=(labelClipedPath, imageClipedPath, labelNameList[listPieceNum*i:listPieceNum*i+listPieceNum],backGroundValue,i))
    processList.append(aProcess)
for i in range(len(processList)):
    processList[i].daemon = True
    processList[i].start()
#设置运行完所有进程才进入下一步
for i in range(len(processList)):
    processList[i].join()
print("check end!")

####转换训练标签通道
print("start 3D to 1D!")
colourMap={(0, 0, 0): 0,
           (255, 255, 255): 1}
labelCliped1DPath=clipPath+'/label1D'
os.makedirs(labelCliped1DPath)
transformDimension(labelClipedPath,colourMap,labelCliped1DPath)

####生成描述文件
print("start make txt!")
txtPath = '../train/list'
if os.path.exists(txtPath):
    shutil.rmtree(txtPath)
os.mkdir(txtPath)
##第只需要一个txt文件train.txt，每一行两句，第一句图片，第二句标签
countImage = 0
with open(txtPath + "/train.txt", "w") as f:
    imageClipedNameList = os.listdir(imageClipedPath)
    for imageClipedName in imageClipedNameList:
        f.write(imageClipedPath +'/'+ imageClipedName +' '+labelCliped1DPath + '/' + imageClipedName[0:-4]+'.png\n')


####启动训练
timeNow=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
netStructure='/home/yqy/computerVison/RSBD4/netVersion/4SV3/train.prototxt'
resultPath = '../train/result'

logPath='../train/log'
if os.path.exists(logPath):
    shutil.rmtree(logPath)
os.mkdir(logPath)

resultPrefix=resultPath+'/'+timeNow

caffeToolPath='/home/yqy/computerVison/deepLabCaffe/build/tools/caffe'
solverFilePath='../netVersion/4SV3/solver.prototxt'

#修改solver里面的内容
lines = open(solverFilePath,'r').readlines()
fLen= len(lines) - 1
for i in range(fLen):
    if "train_net" in lines[i]:
        lines[i] = "train_net:"+"\""+netStructure+"\""+"\n"
    if "snapshot_prefix" in lines[i]:
        lines[i] = "snapshot_prefix:\"" + resultPrefix+ "\""+"\n"
open(solverFilePath, 'w').writelines(lines)

#修改train.prototxt里面的内容
lines = open(netStructure,'r').readlines()
fLen= len(lines) - 1
for i in range(fLen):
    if "root_folder" in lines[i]:
        lines[i] = "root_folder:"+"\"" + "\""+"\n"
        lines[i+1] = "source:\""+txtPath+"/train.txt"+"\""+"\n"
open(netStructure, 'w').writelines(lines)

initFilePath='../initModel/init.caffemodel'
logName=timeNow+'.log'
os.system( caffeToolPath+" train --solver="+solverFilePath+" --weights="+initFilePath+" 2>&1 | tee "+logPath+'/'+logName)