#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'       # 网上随机下载的大象图片
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)     # 图片转成numpy array
x = np.expand_dims(x, axis=0)   # 扩展维度
x = preprocess_input(x)         # 预处理

preds = model.predict(x)
# 将结果解码成一个元组列表（类、描述、概率）（批次中每个样本的一个这样的列表）
print('Predicted:', decode_predictions(preds))
