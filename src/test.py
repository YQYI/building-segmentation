# -*- coding: utf-8 -*-
###############################涉及到图片全是jpg，涉及到标签全是png
import os
import cv2
import shutil
import scipy.io as scio
import numpy as  np



import multiprocessing
cpuNum=multiprocessing.cpu_count()
print("cpu总数："+str(cpuNum))
#overlapSplit这里面有个很大的逻辑陷阱，注意计算有效边长是减去一个重叠因子


def overlapSplit(imagePath,P,netInputSide,savePath):
    suffix=imagePath[-4:]
    image=cv2.imread(imagePath)
    imageH=image.shape[0]
    imageW=image.shape[1]
    #有效边长
    validSide=netInputSide-P
    #针对有效边长的补全
    validePadW=validSide-imageW%validSide
    validePadH =validSide-imageH%validSide
    imageValidPaded=cv2.copyMakeBorder(image,0,validePadH,0,validePadW,cv2.BORDER_CONSTANT, value=0)
    #计算最终结果应有的分块数量
    splitH = int(imageValidPaded.shape[0] / validSide)
    print(splitH)
    splitW = int(imageValidPaded.shape[1] / validSide)
    print(splitW)

    imagePaded=cv2.copyMakeBorder(imageValidPaded,P,P,P,P,cv2.BORDER_CONSTANT, value=0)

    movePace=netInputSide-P
    for i in range(splitH):
        for j in range(splitW):
            print(str(i)+"_"+str(j))
            piece=imagePaded[movePace*i:movePace*i+netInputSide,movePace*j:movePace*j+netInputSide]
            cv2.imwrite(savePath+'/'+str(i)+'_'+str(j)+suffix,piece)
    return [splitH,splitW,validePadH,validePadW]


def transformMatToPng(matList,savePath,processID):
    for mat in matList:
        print("process"+str(processID)+" is transforming "+mat)
        data = scio.loadmat(matPath+'/'+mat)
        imageArray = data['data']
        imageArray = imageArray.transpose(1, 0, 2, 3)
        result = np.argmax(imageArray, axis=2).reshape(321, 321)
        pngImage = np.zeros((321, 321, 3), dtype=np.uint8)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i][j] == 0:
                    pngImage[i][j] = (0, 0, 0)
                else:
                    pngImage[i][j] = (255, 255, 255)
        cv2.imwrite(savePath+'/'+mat[0:-4]+'.png', pngImage)

def mergeResult(piecePath,savePath,P,splitInfo,side):
    vList=[]
    for i in range(splitInfo[0]):
        hList=[]
        for j in range(splitInfo[1]):
            aPath=piecePath+'/'+str(i)+'_'+str(j)+"*"+'_blob_0.png'
            print(aPath)
            piece=cv2.imread(aPath)
            cut=int(P/2)
            validPiece=piece[cut:side-cut,cut:side-cut]
            hList.append(validPiece)
        resultImageH=hList[0]
        for index in range(len(hList)):
            if index==0:
                continue
            else:
                resultImageH=np.concatenate([resultImageH,hList[index]],axis=1)
        vList.append(resultImageH)
    resultImage =vList[0]
    for index in range(len(vList)):
        if index==0:
            continue
        else:
            resultImage=np.concatenate([resultImage,vList[index]],axis=0)
    #recovery size
    resultImage=resultImage[0:resultImage.shape[0]-splitInfo[2],0:resultImage.shape[1]-splitInfo[3]]
    cv2.imwrite(savePath+'/merge.png',resultImage)



####overSplit
imagePath='../data/0818数据/1.jpg'
P=40
netInputSide=321
savePath='../test/overlapSplit'
if os.path.exists(savePath):
    shutil.rmtree(savePath)
os.mkdir(savePath)
splitInfo=overlapSplit(imagePath,P,netInputSide,savePath)


####create txt
txtPath='../test/list'
if os.path.exists(txtPath):
    shutil.rmtree(txtPath)
os.mkdir(txtPath)
##第一个文件用于caffe找到图片检测
countImage=0
with open(txtPath+"/imagePath.txt", "w") as f:
    imagePathList=os.listdir(savePath)
    for imagePath in imagePathList:
        f.write(os.path.dirname(os.path.realpath(__file__))+'/'+savePath+'/'+imagePath+'\n')
        countImage=countImage+1
##第二个文件用于caffe输出mat文件命名
with open(txtPath+"/imageID.txt", "w") as f:
    imagePathList=os.listdir(savePath)
    for imagePath in imagePathList:
        f.write(imagePath[0:-4]+'\n')



####开始调用caffe检测
##文件设置
caffeToolsPath="/home/yqy/computerVison/deeplabV2History/deeplabV2ORI/build/tools/caffe"
iterNum=countImage
model="../train/result/SH4SV3Init.caffemodel"
netStructure="../netVersion/4SV3/test.prototxt"

##先准备mat特征的输出环境solver
matPath='../test/matFeature'
if os.path.exists(matPath):
    shutil.rmtree(matPath)
os.mkdir(matPath)
##将txt路径写到test.prototxt中去,一个是输入文件的txt，还有是mat的txt和位置
lines = open(netStructure,'r').readlines()
fLen= len(lines) - 1
for i in range(fLen):
    if "root_folder" in lines[i]:
        lines[i] = "root_folder:"+"\"" + "\""+"\n"
        lines[i+1] = "source:\""+txtPath+"/imagePath.txt"+"\""+"\n"
    if "prefix" in lines[i]:
        lines[i] = "prefix:\"" + matPath + "/"+"\""+"\n"
        lines[i + 1] = "source:\"" + txtPath+"/imageID.txt" + "\""+"\n"
open(netStructure, 'w').writelines(lines)
##开始检测
os.system( caffeToolsPath+" test --model="
           +netStructure+" --weights="
           +model
           +" --gpu=0 --iterations="
           +str(iterNum))

####transform mat to png

resultPath='../test/result'
if os.path.exists(resultPath):
    shutil.rmtree(resultPath)
os.mkdir(resultPath)

matList=os.listdir(matPath)

processList=[]
pieceSize=int(len(matList)/cpuNum)

for i in range(cpuNum):
    if i==cpuNum-1:
        aProcess = multiprocessing.Process(
            target=transformMatToPng,
            args=(matList[pieceSize * i:], resultPath,i))
    else:
        aProcess=multiprocessing.Process(
            target=transformMatToPng,
            args=(matList[pieceSize*i:pieceSize*(i+1)],resultPath,i))
    processList.append(aProcess)

for i in range(len(processList)):
    processList[i].daemon = True
    processList[i].start()
# 设置运行完所有进程才进入下一步
for i in range(len(processList)):
    processList[i].join()

####拼接结果并且成图,包括还原大小
reportPath='../test/report'
if os.path.exists(reportPath):
    shutil.rmtree(reportPath)
os.mkdir(reportPath)
mergeResult(resultPath,reportPath,P,splitInfo,netInputSide)
