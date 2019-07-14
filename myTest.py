# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:14:47 2019

@author: USER
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 09:02:39 2019
@author: User
"""

""" Installation
pip install keras-segmentation
"""

import keras_segmentation

model = keras_segmentation.pretrained.pspnet_50_ADE_20K() # load the pretrained model trained on ADE20k dataset

#model = keras_segmentation.pretrained.pspnet_101_cityscapes() # load the pretrained model trained on Cityscapes dataset

#model = keras_segmentation.pretrained.pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset

# load any of the 3 pretrained models

out = model.predict_segmentation(
    inp="test.jpg",
    out_fname="out.png"
)

import cv2
img = cv2.imread("test.jpg")
cv2.imshow("Input Image",img)
img = cv2.imread("out.png")
cv2.imshow("Output Image",img)
cv2.waitKey(1000)