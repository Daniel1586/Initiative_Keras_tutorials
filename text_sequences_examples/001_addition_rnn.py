#!/usr/bin/python
# -*- coding: utf-8 -*-

# 序列到序列学习加法运算
"""
An implementation of sequence to sequence learning for performing addition of two numbers.
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be reversed, shown to increase performance in many tasks in:
"Learning to Execute" http://arxiv.org/abs/1410.4615 and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
Two digits reversed:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
Three digits reversed:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
Four digits reversed:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
Five digits reversed:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
"""
# Output after 100 epochs on CPU(i5-7500): ~0.9828

import numpy as np
from keras import layers
from six.moves import range
from keras.models import Sequential


class CharacterTable(object):
    """
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    # 初始化字符表
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """
        One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding.
            This is used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class Colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# Parameters for the model and dataset.
DIGITS = 3
REVERSE = True
TRAINING_SIZE = 50000

# Maximum length of input is 'int + int' (e.g., '345+678').
# Maximum length of int is DIGITS.
MAXLEN = DIGITS + 1 + DIGITS

# All the numbers, plus sign and space for padding.
# chars通过转换形成2个字典序列,1个list
# chars: [' ','+','0','1','2','3','4','5','6','7','8','9']
# char to indices:
# {' ': 0, '+': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11}
# indices to char:
# {0: ' ', 1: '+', 2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9'}
chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()

# 生成测试用例
# while循环将生成加法问题及答案,分别保存在questions和expected中
print('========== 1.Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()

    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)

    # Pad the data with spaces such that it is always MAXLEN.
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)

    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'.
        # (Note the space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('----- Total addition questions:', len(questions))

# 创建x,y,保存问题和答案,x:50000个问题,7位字符串,每一位对应12个字符
print('========== 2.Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)       # shape(50000,7,12)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)   # shape(50000,7,12)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# Shuffle (x, y) in unison as the later parts of x will almost all be larger digits.
# 打散x,y的元素顺序
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over.
# 切分训练集和测试集
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('----- Training Data:')
print(x_train.shape)
print(y_train.shape)

print('----- Validation Data:')
print(x_val.shape)
print(y_val.shape)

# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

# 搭建神经网络模型
print('========== 3.Building model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length, use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))   # 输出(*,128)

# As the decoder RNN's input, repeatedly provide with the last hidden state of RNN for each time step.
# Repeat 'DIGITS + 1' times as that's the maximum length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(DIGITS + 1))  # 输出(*,4,128)

# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but all the outputs
    # so far in the form of (num_samples, timesteps, output_dim). This is necessary as
    # TimeDistributed in the below expects the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))  # 输出(*,4,128)

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars))))     # 输出(*,4,12)
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation dataset.
for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, validation_data=(x_val, y_val))

    # Select 10 samples from the validation set at random so we can visualize errors.
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(Colors.ok + '☑' + Colors.close, end=' ')
        else:
            print(Colors.fail + '☒' + Colors.close, end=' ')
        print(guess)
