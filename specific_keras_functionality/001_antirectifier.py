#!/usr/bin/python
# -*- coding: utf-8 -*-

# 如何在keras中编写自定义层
"""
The example demonstrates how to write custom layers for Keras. We build a custom activation
layer called 'Antirectifier', which modifies the shape of the tensor that passes through it.
We need to specify two methods: `compute_output_shape` and `call`.
Note that the same result can also be achieved via a Lambda layer. Because our custom layer
is written with primitives from the Keras backend (`K`), our code can run both on TensorFlow
and Theano.
"""
# Output after 40 epochs on CPU(i5-7500): ~0.9818

import keras
from keras import layers
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential


# 自定义层Antirectifier
class Antirectifier(layers.Layer):
    """
    This is the combination of a sample-wise L2 normalization with the concatenation of the
    positive part of the input with the negative part of the input. The result is a tensor
    of samples that are twice as large as the input samples.
    It can be used in place of a ReLU.
    # Input shape
        2D tensor of shape (samples, n)
    # Output shape
        2D tensor of shape (samples, 2*n)
    # Theoretical justification
        When applying ReLU, assuming that the distribution of the previous output is
        approximately centered around 0., you are discarding half of your input. This
        is inefficient. Antirectifier allows to return all-positive outputs like ReLU,
        without discarding any data. Tests on MNIST show that Antirectifier allows to
        train networks with twice less parameters yet with comparable classification
        accuracy as an equivalent ReLU-based network.
    """

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2      # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)


# global parameters
batch_size = 128
num_classes = 10
epochs = 40

# the data, split between train and test sets
# MNIST数据集: 训练集60000,测试集10000
print('========== 1.Loading data...')
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
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
model.add(layers.Dense(256, input_shape=(784,)))    # 输出(*,256)
model.add(Antirectifier())                          # 输出(*,512)
model.add(layers.Dropout(0.1))                      # 输出(*,512)
model.add(layers.Dense(256))                        # 输出(*,256)
model.add(Antirectifier())                          # 输出(*,512)
model.add(layers.Dropout(0.1))                      # 输出(*,512)
model.add(layers.Dense(num_classes))                # 输出(*,10)
model.add(layers.Activation('softmax'))             # 输出(*,10)

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
          validation_data=(x_test, y_test))
