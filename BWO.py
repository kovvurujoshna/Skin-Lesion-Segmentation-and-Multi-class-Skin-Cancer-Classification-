import numpy as np
import time


def BWO(SalpPositions, objfun, Lb, Ub, Max_iter):
    #  Beluga whale optimization algorithm
    ub = Ub[1, :]
    lb = Lb[1, :]
    N, dim = SalpPositions.shape
    # initialize alpha, beta, and delta_pos
    FoodPosition = np.zeros((1, dim))
    FoodFitness = np.inf

    Sorted_salps = np.zeros((N, dim))

    SalpFitness = objfun(SalpPositions)
    sorted_salps_fitness, sorted_indexes = np.sort(SalpFitness)
    for newindex in range(N):
        Sorted_salps[newindex, :] = SalpPositions[sorted_indexes[newindex], :]

    FoodPosition = Sorted_salps[1, :]
    FoodFitness = sorted_salps_fitness[1]
    # Initialize the positions of search agents
    # Positions=initialization(SearchAgents_no,dim,ub,lb);

    Convergence_curve = np.zeros((1, Max_iter))
    l = 1
    ct = time.time()


    # Main loop
    while l < Max_iter:
        c1 = 2 * np.exp(- (4 * l / Max_iter) ** 2)
        for i in np.arange(1, SalpPositions.shape[1 - 1] + 1).reshape(-1):
            SalpPositions = np.transpose(SalpPositions)
            if i <= N / 2:
                for j in np.arange(1, dim + 1, 1).reshape(-1):
                    c2 = np.random.rand()
                    c3 = np.random.rand()
                    ############# # Eq. (3.1) in the paper ##############
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition(j) + c1 * ((ub(j) - lb(j)) * c2 + lb(j))
                    else:
                        SalpPositions[j, i] = FoodPosition(j) - c1 * ((ub(j) - lb(j)) * c2 + lb(j))
                    ######################################################
            else:
                if i > N / 2 and i < N + 1:
                    point1 = SalpPositions[:, i - 1]
                    point2 = SalpPositions[:, i]
                    SalpPositions[:, i] = (point2 + point1) / 2
            SalpPositions = np.transpose(SalpPositions)
        for i in np.arange(1, SalpPositions.shape[1 - 1] + 1).reshape(-1):
            Tp = SalpPositions[i, :] > ub
            Tm = SalpPositions[i, :] < lb
            SalpPositions[i, :] = (np.multiply(SalpPositions[i, :], (not (Tp + Tm)))) + np.multiply(ub,
                                                                                                    Tp) + np.multiply(
                lb, Tm)
            SalpFitness[1, i] = objfun(SalpPositions[i, :])
            if SalpFitness[1, i] < FoodFitness:
                FoodPosition = SalpPositions[i, :]
                FoodFitness = SalpFitness(1, i)
        Convergence_curve[1, l] = FoodFitness
        l = l + 1

    bestfit = Convergence_curve[1, Convergence_curve.shape[1]-1]
    ct = time.time() - ct
    return bestfit, Convergence_curve, FoodPosition, ct
