import random as rn
import math
from math import ceil, sin, exp, pi
import numpy as np
from numba import double
from torch import numel
import time


# Enhanced Cheetah Optimizer (CO)
def CO(Positions, fobj, VRmin, VRmax, Max_iter):
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    BestCost = np.zeros((dim, 1))
    BestSol = float('inf')
    Globest = BestCost

    for i in range(N):
        Position = lb + np.random.rand(1, dim) * (ub - lb)
        Cost = fobj(Position)
        if Cost < BestSol.Cost:
            BestSol = BestCost
    Convergence_curve = np.zeros((Max_iter, 1))
    pop1 = BestCost
    BestCost = []
    X_best = BestSol
    Globest = BestCost
    it = 1
    T = ceil(Positions / 10) * 60
    FEs = 0

    t = 0
    ct = time.time()
    while FEs <= Max_iter:
        dl = np.random.rand(dim, N, 1)
        for k in range(dim):
            i = dl[k]

            if k == len(dl):
                a = dl[k - 1]
            else:
                a = dl[k + 1]

            X = Positions
            X1 = Positions[a]
            Xb = BestSol.Position
            Xbest = BestSol.Position
            kk = 0

            xd = np.zeros(X)
            if (it % 100) == 0 or it == 1:
                xd = np.random.permutation(numel(X))
            Z = X
            for j in range(xd):
                r_Hat = rn.random()
                r1 = rn.random()
                if k == 1:
                    alpha = 0.0001 * t / T * (ub[j] - lb[j])
                else:
                    alpha = 0.0001 * t / T * abs(Xb[j] - X[j]) + 0.001 * round(double(rn.random() > 0.9))

                r = rn.random()
                r_Check = abs(r) ** exp(r / 2) * sin(2 * pi * r)
                beta = X1[j] - X[j]

                h0 = exp(2 - 2 * t / T)

                H = abs(2 * r1 * h0 - h0)

                r2 = rn.random()
                r3 = kk + rn.random()

                if r2 <= r3:
                    r4 = 3 * rn.random()
                    if H > r4:
                        Z[j] = X[j] + r_Hat ** -1 * alpha
                    else:
                        Z[j] = Xbest[j] + r_Check * beta
                else:
                    Z[j] = X[j]

            xx1 = np.var[Z < lb]
            Z[xx1] = lb(xx1) + np.random.rand(numel(xx1)) * (ub(xx1) - lb(xx1))
            xx1 = np.var[Z > ub]
            Z[xx1] = lb(xx1) + np.random.rand(numel(xx1)) * (ub(xx1) - lb(xx1))

            Positions = Z
            fitness = fobj(Positions[i, :])
            if fitness < BestSol:
                BestCost = fitness  # Update alpha
                BestSol = Positions[i, :]

            FEs = FEs + 1

        t = t + 1

        if t > T & t - round(T) - 1 >= 1 & t > 2:
            if abs(BestCost[t - 1] - BestCost[(t - round(T) - 1)]) <= abs(0.01 * BestCost[t - 1]):
                best = BestSol.Position
                jn = np.random.rand(Positions, 1, ceil(Positions / 10 * rn.random()))
                best[jn] = lb(jn) + np.random.rand(1, len(jn)) * (ub(jn) - lb(jn))
                BestSol.Cost = fobj(best)
                BestSol.Position = best
                FEs = FEs + 1

                i0 = np.random.rand(N, 1, round(1 * N))

                BestCost[i0[N - dim + 1: N]] = pop1[dl[1: dim]]

                BestCost = X_best

                t = 1

        it = it + 1
        if BestSol < X_best:
            X_best = BestSol
        BestCost[t] = BestSol
        Globest[t] = X_best

        Convergence_curve[t] = X_best
        t = t + 1
    X_best = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return X_best, Convergence_curve, Globest, ct
