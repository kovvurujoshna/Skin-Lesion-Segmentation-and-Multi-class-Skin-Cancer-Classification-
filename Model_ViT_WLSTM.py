
from Model_LSTM import Model_LSTM
from Model_Vision_Transformer import Model_Vision_Transformer


def Model_ViT_WLSTM(image, Targets, Optimizer=None, sol=None):
    if Optimizer is None:
        Optimizer = 'Adam'
    if sol is None:
        sol = [5, 5, 300]
    Feature = Model_Vision_Transformer(image, Optimizer)
    Eval, Pred, Weight = Model_LSTM(Feature, Targets, Optimizer, sol)
    return Eval, Pred, Weight