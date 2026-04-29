import h5py
import numpy as np
import os
import cv2 as cv
import pandas as pd
from numpy import matlib
# from tf_keras.src.utils import to_categorical
from BWO import BWO
from CO import CO
from GOA import GOA
from Global_Vars import Global_Vars
from MAO import MAO
from Model_Bi_RNN_GRU import Model_Bi_RNN_GRU
from Model_LSTM import Model_LSTM, Model_WLSTM
from Model_RNN import Model_RNN
from Model_RNN_GRU import Model_RNN_GRU
from Model_TransUnetPlusPlus import Model_TransUnetPlusPlus
from Model_ViT_WLSTM import Model_ViT_WLSTM
from Image_Results import *
from Objective_Function import Obj_fun, Obj_fun_CLS
from Proposed import Proposed
from Plot_Results import *

no_of_dataset = 2


def ReadText(filename):
    f = open(filename, "r")
    lines = f.readlines()
    Tar = []
    fileNames = []
    for lineIndex in range(len(lines)):
        if lineIndex and '||' in lines[lineIndex]:
            line = [i.strip() for i in lines[lineIndex].strip().strip('||').replace('||', '|').split('|')]
            fileNames.append(line[0])
            Tar.append(int(line[2]))
    Tar = np.asarray(Tar)
    uniq = np.unique(Tar)
    Target = np.zeros((len(Tar), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(Tar == uniq[i])
        Target[index, i] = 1
    return fileNames, Target


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


def Read_Images(Directory):
    Images = []
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        print(i)
        filename = Directory + out_folder[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images


def Read_Datset_PH2(Directory, fileNames):
    Images = []
    GT = []
    folders = os.listdir(Directory)
    for i in range(len(folders)):
        if folders[i] in fileNames:
            image = Read_Image(Directory + folders[i] + '/' + folders[i] + '_Dermoscopic_Image/' + folders[i] + '.bmp')
            gt = Read_Image(Directory + folders[i] + '/' + folders[i] + '_lesion/' + folders[i] + '_lesion.bmp')
            Images.append(image)
            GT.append(gt)
    return Images, GT


def Read_CSV(Path):
    df = pd.read_csv(Path)
    values = df.to_numpy()
    value = values[:, 6]
    uniq = np.unique(value)
    Target = np.zeros((len(value), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(value == uniq[i])
        Target[index, i] = 1
    return Target


# Read Datasets
an = 0
if an == 1:
    Images1 = Read_Images('./Datasets/HAM10000/Images/')
    np.save('Images_1.npy', Images1)

    Target1 = Read_CSV('./Datasets/HAM10000/HAM10000_metadata.csv')
    np.save('Target_1.npy', Target1)

    fileNames, Target2 = ReadText('./Datasets/PH2Dataset/PH2_dataset.txt')
    Images2, GT = Read_Datset_PH2('./Datasets/PH2Dataset/PH2 Dataset images/', fileNames)
    np.save('Images_2.npy', Images2)
    np.save('GT_2.npy', GT)
    np.save('Target_2.npy', Target2)

# GroundTruth for Dataset1
an = 0
if an == 1:
    im = []
    img = np.load('Images_1.npy', allow_pickle=True)
    for i in range(len(img)):
        print(i)
        image = img[i]
        minimum = int(np.min(image))
        maximum = int(np.max(image))
        Sum = ((minimum + maximum) / 2)
        ret, thresh = cv.threshold(image, Sum, 255, cv.THRESH_BINARY_INV)
        im.append(thresh)
    np.save('GT_1.npy', im)

# Generate Target for Dataset 1
an = 0
if an == 1:
    for n in range(1):
        Tar = []
        Ground_Truth = np.load('GT_1.npy', allow_pickle=True)
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            result = image.astype('uint8')
            uniq = np.unique(result)
            if len(uniq) > 1:
                Tar.append(1)
            else:
                Tar.append(0)
        Tar = (to_categorical(np.asarray(Tar).reshape(-1, 1))).astype('int')
        np.save('Target_' + str(n + 1) + '.npy',Tar)


# Optimization for Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('GT_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.GT = GT
        Npop = 10
        Chlen = 3  # Here we optimized Hidden Neuron Count, No of epoches, Actiation Function
        xmin = matlib.repmat([5, 5, 1], Npop, 1)
        xmax = matlib.repmat([255, 50, 5], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun
        Max_iter = 50

        print("MAO...")
        [bestfit1, fitness1, bestsol1, time1] = MAO(initsol, fname, xmin, xmax, Max_iter)  # MAO

        print("BWO...")
        [bestfit2, fitness2, bestsol2, time2] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

        print("CO...")
        [bestfit3, fitness3, bestsol3, time3] = CO(initsol, fname, xmin, xmax, Max_iter)  # CO

        print("GOA...")
        [bestfit4, fitness4, bestsol4, time4] = GOA(initsol, fname, xmin, xmax, Max_iter)  # GOA

        print("Improved GOA...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Improved GOA

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        np.save('BestSol_seg_' + str(n + 1) + '.npy', BestSol)

# Segmentation
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('GT_' + str(n + 1) + '.npy', allow_pickle=True)
        Bestsol = np.load('BestSol_seg' + str(n + 1) + '.npy', allow_pickle=True)
        sol = np.round(Bestsol[4, :]).astype(np.int16)
        Segmented_Images, Eval = Model_TransUnetPlusPlus(Images, GT, sol)
        np.save('Method5_Dataset' + str(n + 1) + '.npy', Segmented_Images)


# Get weights from ViT_LSTM
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Method5_Dataset' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target' + str(n + 1) + '.npy', allow_pickle=True)
        Eval, Pred, mod_1 = Model_ViT_WLSTM(Images, Target)
        with h5py.File('model.h5', 'w') as f:
            f.create_dataset("LSTM", data=mod_1)


# Optimization for Weighted Long Short Term Memory (ViT-WLSTM)
an = 0
if an == 1:
    for n in range(no_of_dataset):
        with h5py.File('model.h5', 'r') as file:
            weight = file['LSTM'][:]
        Images = np.load('Method5_Dataset' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.Target = Tar
        Global_Vars.Weight = weight
        Npop = 10
        Chlen = 3  # Here we optimized Hidden Neuron Count, No of epoches, Steps per epoch in Transunet3+
        xmin = matlib.repmat([5, 5, 300], Npop, 1)
        xmax = matlib.repmat([255, 50, 1000], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Obj_fun_CLS
        Max_iter = 25

        print("MAO...")
        [bestfit1, fitness1, bestsol1, time1] = MAO(initsol, fname, xmin, xmax, Max_iter)  # MAO

        print("BWO...")
        [bestfit2, fitness2, bestsol2, time2] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

        print("CO...")
        [bestfit3, fitness3, bestsol3, time3] = CO(initsol, fname, xmin, xmax, Max_iter)  # CO

        print("GOA...")
        [bestfit4, fitness4, bestsol4, time4] = GOA(initsol, fname, xmin, xmax, Max_iter)  # GOA

        print("Improved GOA...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Improved GOA

        BestSol=[bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

        np.save('BestSol_' + str(n + 1) + '.npy', BestSol)

# Classification
an = 0
if an == 1:
    Evaluate_all = []
    for n in range(no_of_dataset):
        with h5py.File('model.h5', 'r') as file:
            weight = file['LSTM'][:]
        Feat = np.load('Method5_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        Tar = np.load('Target_'+str(n+1)+'.npy', allow_pickle=True)
        bests = np.load('BestSol_'+str(n+1)+'.npy', allow_pickle=True)
        Eval_all = []
        Optimizer = ['Adam', 'SGD', 'RMSProp', 'Adadelta', 'AdaGrad']
        for m in range(len(Optimizer)):  # for all learning percentage
            EVAL = np.zeros((10, 14))
            per = round(len(Feat) * 0.75)
            Train_Data = Feat[:per, :, :]
            Train_Target = Tar[:per, :]
            Test_Data = Feat[per:, :, :]
            Test_Target = Tar[per:, :]
            for j in range(bests.shape[0]):  # for all algorithms
                soln = bests[j]
                EVAL[j, :], pred = Model_WLSTM(weight, Tar, weight, Optimizer, soln)  # with Optimization ViT with Weighted LSTM
            EVAL[5, :], pred1 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer)  # Resnet Model
            EVAL[6, :], pred2 = Model_RNN_GRU(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer)  # Inception model
            EVAL[7, :], pred3 = Model_Bi_RNN_GRU(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer)  # Mobilenet model
            EVAL[8, :], pred4 = Model_WLSTM(weight, Tar, weight, Optimizer)  # without Optimization ViT with Weighted LSTM
            EVAL[9, :] = EVAL[4, :]
            Eval_all.append(EVAL)
        Evaluate_all.append(Eval_all)
    np.save('Eval_all.npy', Evaluate_all)


plot_results_optimizer()
plot_results()
plot_Segmentation_results_1()
plotConvResults()
Plot_ROC_Curve()
Image_Results()
Sample_Images()