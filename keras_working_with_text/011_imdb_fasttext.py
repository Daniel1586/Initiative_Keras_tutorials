#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This example demonstrates the use of fasttext for text classification Based on Joulin et al's paper:
Bags of Tricks for Efficient Text Classification. https://arxiv.org/abs/1607.01759
Results on IMDB datasets with uni and bi-gram embeddings:
    Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
    Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
"""
# 利用fasttext模型对IMDB影评倾向分类
# Uni-gram: Output after 5 epochs on CPU(i5-7500): ~0.8878

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.datasets import imdb


def create_ngram_set(inputlist, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[inputlist[k:] for k in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for inputlist in sequences:
        new_list = inputlist[:]
        for ngram_value in range(2, ngram_range + 1):
            for k in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[k:k + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


# Set parameters:
# ngram_range = 2 will add bi-grams features
ngram_range = 1
max_features = 20000
maxlen = 400
batch_size = 32
embedding_dims = 50
epochs = 5

# 数据集来源IMDB影评,共50000条影评,标记正面/负面两种评价
# 每条数据被编码为一条索引序列(索引数字越小,代表单词出现次数越多)
# num_words: 选取的每条数据里的索引值不能超过num_words
print('========== 1.Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print('----- train sequences', len(x_train))
print('----- test  sequences', len(x_test))
print('----- Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
print('----- Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

#
if ngram_range > 1:
    print('----- Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('----- Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
    print('----- Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))

# 对每条词索引组成的数据进行长度对齐,去掉数据前面或后面多余的单词;长度不够插入0
print('========== 2.Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('----- x_train shape:', x_train.shape)
print('----- x_test shape:', x_test.shape)

# 搭建神经网络模型
print('========== 3.Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))     # 输出(*,400,50)

# we add a GlobalAveragePooling1D, which will average the embeddings
# of all words in the document
model.add(GlobalAveragePooling1D())     # 输出(*,50)

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1, activation='sigmoid'))   # 输出(*,1)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
