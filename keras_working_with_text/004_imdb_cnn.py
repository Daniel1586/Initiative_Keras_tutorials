#!/usr/bin/python
# -*- coding: utf-8 -*-

# This example demonstrates the use of Convolution1D for text classification.
# 利用1D-CNN模型对IMDB影评倾向分类

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb

max_features = 5000     # vocab大小
maxlen = 400            # 每条样本数据长度
batch_size = 32         # min-batch size
embedding_dims = 50     # 词向量维度
filters = 250           # 1D-CNN卷积核的数目(即输出的维度)
kernel_size = 3         # 整数或由单个整数构成的list/tuple,卷积核的空域或时域窗长度
hidden_dims = 250       # 隐藏层神经元数量
epochs = 5              # 循环次数

# 数据集来源IMDB影评,共50000条影评,标记正面/负面两种评价
# 每条数据被编码为一条索引序列(索引数字越小,代表单词出现次数越多)
# num_words: 选取的每条数据里的索引值不能超过num_words
print('========== 1.Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print('----- train sequences', len(x_train))
print('----- test  sequences', len(x_test))

# 对每条词索引组成的数据进行长度对齐,去掉数据前面或后面多余的单词;长度不够插入0
print('========== 2.Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('----- x_train shape:', x_train.shape)
print('----- x_test shape:', x_test.shape)

# 搭建神经网络模型
print('========== 3.Build model...')
model = Sequential()

# input_dim=max_features单词表大小,output_dim=embedding_dims=50为词向量维度,input_length=maxlen每条样本数据长度
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))     # 输出(*,400,50)
model.add(Dropout(0.2))

# 1维卷积层,卷积输出维度为filters,卷积步长为strides
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))  # 输出(*,398,250)
# 对于时间信号的全局最大池化
model.add(GlobalMaxPooling1D())     # 输出(*,250)

model.add(Dense(hidden_dims))       # 输出(*,250)
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# 神经网络编译/训练/测试集测试性能
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
