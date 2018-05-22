#!/usr/bin/python
# -*- coding: utf-8 -*-

# Train a recurrent convolutional network on the IMDB sentiment classification task.
# 利用1D-CNN-LSTM模型对IMDB影评倾向分类
# Output after 5 epochs on CPU(i5-7500): ~0.8309

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

# Embedding
max_features = 20000        # vocab大小
maxlen = 100                # 每条样本数据长度
embedding_size = 128        # 词向量维度

# Convolution
kernel_size = 5             # 整数或由单个整数构成的list/tuple,卷积核的空域或时域窗长度
filters = 64                # 1D-CNN卷积核的数目(即输出的维度)
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30             # min-batch size
epochs = 5                  # 循环次数

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

# input_dim=max_features单词表大小,output_dim=embedding_dims=128为词向量维度,input_length=maxlen每条样本数据长度
model.add(Embedding(max_features, embedding_size, input_length=maxlen))     # 输出(*,100,128)
model.add(Dropout(0.25))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))  # 输出(*,96,64)
model.add(MaxPooling1D(pool_size=pool_size))  # 输出(*,24,64)
model.add(LSTM(lstm_output_size))             # 输出(*,70)
model.add(Dense(1))
model.add(Activation('sigmoid'))

# 神经网络编译/训练/测试集测试性能
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('----- Test score:', score)
print('----- Test accuracy:', acc)
