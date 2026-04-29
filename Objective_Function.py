from numpy.random import rand
from Global_Vars import Global_Vars
import numpy as np
from Model_LSTM import Model_WLSTM
from Model_TransUnetPlusPlus import Model_TransUnetPlusPlus


def Obj_fun(Soln):
    Images = Global_Vars.Images
    GT = Global_Vars.GT
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 1:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i]).astype('uint8')
            Eval, Segimg = Model_TransUnetPlusPlus(Images, GT, sol)
            Fitn[i] = 1 / Eval[4]
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        Eval, Segimg = Model_TransUnetPlusPlus(Images, GT, sol)
        Fitn = 1 / Eval[4]
        return Fitn


def Obj_fun_CLS(Soln):
    Feat = Global_Vars.Images
    Target = Global_Vars.Target
    weight = Global_Vars.weight
    learnperc = Global_Vars.learnperc
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 1:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i]).astype('uint8')
            Eval = Model_WLSTM(Feat, Target, weight, sol)
            Fitn[i] = 1 / Eval[4]
        return Fitn
    else:
        sol = np.round(Soln).astype('uint8')
        Eval = Model_WLSTM(Feat, Target, weight, sol)
        Fitn = 1 / Eval[4]
        return Fitn