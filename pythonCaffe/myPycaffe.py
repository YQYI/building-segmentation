# -*- coding: utf-8 -*-
import numpy as np
import sys,os
import cv2


import sys
caffe_root = '/home/yqy/computerVison/deepLabCaffe/'
sys.path.insert(0, caffe_root+'python')
import caffe
netStructure="../netVersion/4SV3/test.prototxt"
model="../train/result/SH4SV3Init.caffemodel"
mean_file=caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'




net=caffe.Net(netStructure,model,caffe.TEST)

out=net.forward()#进行向前传播，因为是测试阶段
