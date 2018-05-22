#!/usr/bin/python
# -*- coding: utf-8 -*-

# Trains and evaluate a simple MLP on the Reuters newswire topic classification task.
# 训练并评估一个简单的MLP(对路透社新闻主题分类)

import keras
import numpy as np
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

max_words = 1000    # vocab大小
batch_size = 32     # min-batch size
epochs = 5          # 循环次数

# 数据集来源路透社新闻专线,共11228条新闻,标记46个类别
# 每条数据被编码为一条索引序列(索引数字越小,代表单词出现次数越多)
# num_words: 选取的每条数据里的索引值不能超过num_words
# test_split: test data所占数据集比例
print('========== 1.Loading data...')
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.2)
print('----- train sequences', len(x_train))
print('----- test  sequences', len(x_test))
num_classes = np.max(y_train) + 1
print('----- classes num', num_classes)

# 对每条词索引组成的数据(train/test)转换为词典长度的0/1值序列(one-hot),若单词出现则为1,否则为0
print('========== 2.Vectorizing sequence data...')
tokenizer = Tokenizer(num_words=max_words)  # 只记录max_words数量的单词信息
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print('----- x_train shape:', x_train.shape)
print('----- x_test  shape:', x_test.shape)

# 对每条数据的类别标签(train/test)转换为类别数目的0/1值序列(one-hot)
print('Convert class vector to binary class matrix ''(for use with categorical_crossentropy)')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('----- y_train shape:', y_train.shape)
print('----- y_test  shape:', y_test.shape)

# 搭建神经网络模型
print('========== 3.Building model...')
model = Sequential()
# 第一层
model.add(Dense(512, input_shape=(max_words,)))     # 输入(*,max_words), 输出(*,512)
model.add(Activation('relu'))                       # 输出(*,512)
model.add(Dropout(0.5))                             # 输出(*,512)
# 第二层
model.add(Dense(num_classes))                       # 输出(*,num_classes)
model.add(Activation('softmax'))                    # 输出(*,num_classes)

# 损失函数设置,优化函数设置,模型评估性能指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 神经网络训练和交叉验证模型性能
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)
# 测试集性能测试
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print('----- Test loss:', score[0])
print('----- Test accuracy:', score[1])
