#!/usr/bin/python
# -*- coding: utf-8 -*-

# 训练CNN模型对MNIST数据集分类
# Trains a simple convnet on the MNIST dataset. Gets to 99.25% test accuracy after 12 epochs
# (there is still a lot of margin for parameter tuning). 16 seconds per epoch on a GRID K520 GPU.
# Output after 12 epochs on CPU(i5-7500)/GPU(1050Ti): ~0.9918

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128    # min-batch size
num_classes = 10    # 类别数
epochs = 12         # 循环次数

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# MNIST数据集: 训练集60000,测试集10000
print('========== 1.Loading data...')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 后端使用tensorflow时,通道维在最后
# (100,28,28,3): 100是样本维,28*28是图片维度,3是通道维,表示颜色通道数
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 归一化
x_test /= 255
print('----- x_train shape:', x_train.shape)
print('----- x_test  shape:', x_test.shape)

# convert class vectors to binary class matrices
# 对每条数据的类别标签(train/test)转换为类别数目的0/1值序列(one-hot)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 搭建神经网络模型
print('========== 2.Building model...')
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))   # 输出(*,26,26,32)
model.add(Conv2D(64, (3, 3), activation='relu'))    # 输出(*,24,24,64)
model.add(MaxPooling2D(pool_size=(2, 2)))           # 输出(*,12,12,64)
model.add(Dropout(0.25))                            # 输出(*,12,12,64)
model.add(Flatten())                                # 输出(*,9216)
model.add(Dense(128, activation='relu'))            # 输出(*,128)
model.add(Dropout(0.5))                             # 输出(*,128)
model.add(Dense(num_classes, activation='softmax'))     # 输出(*,10)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('----- Test loss:', score[0])
print('----- Test accuracy:', score[1])
