import numpy as np
from keras import Sequential
from keras.src.layers import LSTM, Dense, Bidirectional

from Evaluation import evaluation


# https://www.tensorflow.org/guide/keras/rnn


def Model_RNN(train_data, train_target, test_data, test_target, Steps_per_Epochs, sol=50):
    pred, model = RNN_train(train_data, train_target, test_data, Steps_per_Epochs, sol)  # RNN
    pred = np.squeeze(pred)

    Eval = evaluation(pred, test_target)
    return Eval, pred


def RNN_train(trainX, trainY, testX, Steps_per_Epochs, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol), input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, steps_per_epoch=Steps_per_Epochs, batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return testPredict, model

def Model_BiRNN(train_data, train_target, test_data, test_target, Steps_per_Epochs, sol=50):
    pred, model = BiRNN_train(train_data, train_target, test_data, Steps_per_Epochs, sol)  # RNN
    pred = np.squeeze(pred)

    Eval = evaluation(pred, test_target)
    return Eval, pred


def BiRNN_train(trainX, trainY, testX, Steps_per_Epochs, sol):
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(int(sol), input_shape=(1, trainX.shape[2])))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=2, steps_per_epoch=Steps_per_Epochs, batch_size=1, verbose=2)
    # make predictions
    # trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return testPredict, model