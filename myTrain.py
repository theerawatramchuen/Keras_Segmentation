# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:01:53 2019

@author: USER
"""

import keras_segmentation

model = keras_segmentation.models.unet.vgg_unet(n_classes=51 ,  input_height=416, input_width=608  )

model.train( 
    train_images =  "dataset1/images_prepped_train",
    train_annotations = "dataset1/annotations_prepped_train",
    checkpoints_path = "tmp/vgg_unet_1" , epochs=5
)

out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="tmp/out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)
import cv2
import numpy
inImg = numpy.asarray(cv2.imread("dataset1/images_prepped_test/0016E5_07965.png"))
plt.imshow(inImg)