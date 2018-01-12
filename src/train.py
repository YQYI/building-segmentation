import os
import cv2
import shutil
import numpy as np
import time
from PIL import Image
from datetime import datetime
from multiprocessing import cpu_count
import multiprocessing
from numba import jit
cpuNum=cpu_count()
#图像反色
def reverse(imagePath):
    image=cv2.imread(imagePath)
    img_base=np.ones_like(image,np.uint8)
    img_base=255*img_base
    result=img_base-image
    cv2.imwrite(imagePath,result)


def rotateDataSet(imagePath,savepath,rotateTime):
    image=Image.open(imagePath)
    imageName=imagePath.split('/')[-1]
    aAngle=180//rotateTime
    for i in range(rotateTime):
        image_rotated=image.rotate(180+i*aAngle)
        image_rotated.save(savepath+'/'+str(i*aAngle)+imageName)

def clipImage(imageClipedPath,savePath,side,backGroundValue):
    image=cv2.imread(imageClipedPath)
    padH=int(side-(image.shape[0]/side))
    padW=int(side-(image.shape[1]/side))
    imagePaded=cv2.copyMakeBorder(image, 0, padH, 0, padW, cv2.BORDER_CONSTANT, value=backGroundValue)
    splitH=int(imagePaded.shape[0]/side)
    splitW=int(imagePaded.shape[1]/side)
    for i in range(splitH):
        for j in range(splitW):
            piece=imagePaded[side*i:side*i+side,side*j:side*j+side]
            cv2.imwrite(savePath+'/'+str(i)+'_'+str(j)+imageClipedPath.split('/')[-1],piece)


def ifNeedDelete(labelClipedPath,imageClipedPath,labelList,backGroundValue,num):
    judger = cv2.imread(labelClipedPath + '/' + labelList[0])
    judger0=np.zeros_like(judger,dtype=np.uint8)
    judger255=255*np.ones_like(judger,dtype=np.uint8)
    for labelName in labelList:
        print("process "+str(num)+" is checking " + labelName)
        labelImage = cv2.imread(labelClipedPath + '/' + labelName)
        if (judger0==labelImage).all() or (judger255==labelImage).all():
            os.remove(labelClipedPath + '/' + labelName)
            os.remove(imageClipedPath + '/' + labelName[0:-9] + 'train.png')
        else:
            continue


def transformDimension(labelClipedPath,colourMap,labelCliped1DPath):
    labelClipedNameList=os.listdir(labelClipedPath)
    for aName in labelClipedNameList:
        image=cv2.imread(labelClipedPath+'/'+aName)
        image1D = np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
        for c, i in colourMap.items():
            m = np.all(image == np.array(c).reshape(1, 1, 3), axis=2)
            image1D[m] =i
        cv2.imwrite(labelCliped1DPath+'/'+aName,image1D)

def myMakeDir(dir,ifRewrite=False):
    if os.path.exists(dir):
        if(ifRewrite==False):
            print("dir exists and do nothing!")
            return
        else:
            print("dir exists and delete it!")
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)

##########所有的路径
def starTrainning():
    ####所有的变量设置
    backGroundValue = (0, 0, 0)
    imageSource = '../data/toServer0109/train.png'
    labelSource = '../data/toServer0109/label.png'
    imageRotatePath = '../train/rotate/image'
    labelRotatePath = '../train/rotate/label'
    clipPath = '../train/clipResult'
    imageClipedPath = '../train/clipResult/image'
    labelClipedPath = '../train/clipResult/label'
    labelCliped1DPath = '../train/clipResult/label1D'
    txtPath = '../train/list'

    ####生成目录
    myMakeDir(imageRotatePath, True)
    myMakeDir(labelRotatePath, True)
    myMakeDir(clipPath, True)
    myMakeDir(imageClipedPath, True)
    myMakeDir(labelClipedPath, True)
    myMakeDir(labelCliped1DPath, True)
    myMakeDir(txtPath, True)
    ####旋转训练样本
    print("start clip")
    side=161
    start=datetime.now()
    print("rotate image")
    imageProcess=multiprocessing.Process(target=rotateDataSet,args=(imageSource,imageRotatePath,10))
    print("rotate label")
    labelProcess = multiprocessing.Process(target=rotateDataSet, args=(labelSource, labelRotatePath,10))
    imageProcess.daemon=True
    imageProcess.start()
    labelProcess.daemon=True
    labelProcess.start()
    imageProcess.join()
    labelProcess.join()

    stop = datetime.now()
    print(stop-start)

    ####切割训练样本
    imageRotateList=os.listdir(imageRotatePath)
    labelRotateList=os.listdir(labelRotatePath)
    for aImage in imageRotateList:
        print("clip train image")
        clipImage(imageRotatePath+'/'+aImage,imageClipedPath,side,backGroundValue)

    for aLabel in labelRotateList:
        print("clip label image")
        clipImage(labelRotatePath+'/'+aLabel,labelClipedPath,side,backGroundValue)
    start=datetime.now()
    ####剔除无用样本
    print("start check!")
    labelNameList=os.listdir(labelClipedPath)

    print("共有"+str(cpuNum)+"个cpu")
    listPieceNum=int(len(labelNameList)/cpuNum)
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
    stop = datetime.now()
    print (stop-start)

    ####转换训练标签通道
    print("start 3D to 1D!")
    colourMap={(255, 255, 255): 0,
               (0, 0, 0): 1}
    transformDimension(labelClipedPath,colourMap,labelCliped1DPath)
    ####生成描述文件
    print("start make txt!")

    ##第只需要一个txt文件train.txt，每一行两句，第一句图片，第二句标签
    countImage = 0
    with open(txtPath + "/train.txt", "w") as f:
        imageClipedNameList = os.listdir(imageClipedPath)
        for imageClipedName in imageClipedNameList:
            f.write(imageClipedPath +'/'+ imageClipedName +' '+labelCliped1DPath + '/' + imageClipedName[0:-9]+'label.png\n')


    ####启动训练
    timeNow=time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
    netStructure='/home/yqy/computerVison/RSBD4/netVersion/4SV3/train.prototxt'
    resultPath = '../train/result'
    logPath='../train/log'
    resultPrefix=resultPath+'/'+timeNow
    caffeToolPath='/home/yqy/computerVison/deeplabCaffe/build/tools/caffe'
    solverFilePath='../netVersion/20180111/solver.prototxt'
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

    initFilePath='../train/result/2018-01-11-15-34-12_iter_105000.caffemodel'
    logName=timeNow+'.log'
    os.system( caffeToolPath+" train --solver="+solverFilePath+" --weights="+initFilePath+" 2>&1 | tee "+logPath+'/'+logName)

def testCode():
    ####先切割训练样本
    print("start clip")
    backGroundValue=(0,0,0)
    imageSource='../data/toServer0109/train.png'
    labelSource='../data/toServer0109/label.png'

    side=161
    start=datetime.now()
    imageRotatePath = '../train/rotate/image'
    labelRotatePath = '../train/rotate/label'
    myMakeDir(imageRotatePath, True)
    myMakeDir(labelRotatePath, True)
    print("rotate image")
    rotateDataSet(imageSource,imageRotatePath,10)
    print("rotate label")
    rotateDataSet(labelSource,labelRotatePath,10)
    stop=datetime.now()
    print(stop-start)

    clipPath='../train/clipResult'
    myMakeDir(clipPath, True)

    imageClipedPath='../train/clipResult/image'
    labelClipedPath='../train/clipResult/label'
    myMakeDir(imageClipedPath,True)
    myMakeDir(labelClipedPath,True)
    imageRotateList=os.listdir(imageRotatePath)
    labelRotateList=os.listdir(labelRotatePath)
    for aImage in imageRotateList:
        print("clip train image")
        clipImage(imageRotatePath+'/'+aImage,imageClipedPath,side,backGroundValue)

    for aLabel in labelRotateList:
        print("clip label image")
        clipImage(labelRotatePath+'/'+aLabel,labelClipedPath,side,backGroundValue)
    start=datetime.now()
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
    stop = datetime.now()
    print (stop-start)

    ####转换训练标签通道
    print("start 3D to 1D!")
    colourMap={(255, 255, 255): 0,
               (0, 0, 0): 1}
    labelCliped1DPath='../train/clipResult/label1D'
    myMakeDir(labelCliped1DPath,True)
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
            f.write(imageClipedPath +'/'+ imageClipedName +' '+labelCliped1DPath + '/' + imageClipedName[0:-9]+'label.png\n')

    return 0

starTrainning()
# myMat=np.arange(0,100,5).reshape(2,2,5)
# print (myMat)
# k=myMat.swapaxes(2,1)
# print (k)