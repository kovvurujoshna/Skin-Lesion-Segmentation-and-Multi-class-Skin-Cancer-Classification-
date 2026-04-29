from Model_GRU import Model_GRU
from Model_RNN import Model_RNN


def Model_RNN_GRU(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer):
    if Optimizer is None:
        Optimizer = 'Adam'
    Eval, Pred_RNN = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer)
    Eval, Pred_GRU = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer)
    Pred = (Pred_RNN+Pred_GRU)/2
    return Eval, Pred