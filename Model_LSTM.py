import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from Evaluation import evaluation


def Model_LSTM(Data, Target, weight, Activation_Function, sol=None):
    if sol is None:
        sol=50
    model, weight = LSTM_train(Data, Target, Activation_Function, sol)
    pred = model.predict(Data)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Target)
    return Eval, pred, weight


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(Data, Target, Activation_Function, sol):
    trainX = np.reshape(Data, (Data.shape[0], 1, Data.shape[1]))
    # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(15, input_shape=(1, trainX.shape[2])))
    model.add(Dense(Target.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, Target, epochs=5, batch_size=1, verbose=2)
    weights = model.layers[1].get_weights()
    return weights, model

def Model_WLSTM(Data, Target, weight, Activation_Function, sol=None):
    if sol is None:
        sol = [-20, 20]
    Weight, model = LSTM_train(Data, Target, Activation_Function, sol)
    w = Weight + Weight * (sol[0])
    model = weight

    pred = model.predict(w)
    pred = np.asarray(pred)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Target)
    return Eval, pred



