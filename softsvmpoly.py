import numpy as np
import pandas as pd
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt
import matplotlib.patches as plti
from matplotlib.colors import ListedColormap

from softsvm import softsvm


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    param l: the parameter lambda of the soft SVM algorithm
    param k: the bandwidth parameter sigma of the RBF kernel.
    param trainX: numpy array of size (m, d) containing the training sample
    param trainy: numpy array of size (m, 1) containing the labels of the training sample
    return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    G = retGramMatrix(trainX, trainX, k)
    solvers.options['show_progress'] = False
    G_for_H = 2 * l * G
    m = len(trainy)
    # d = len(trainX[0])

    HRight = np.zeros((2 * m, m))
    zeroLeftCorner = np.zeros((m, m))
    HLeft = np.concatenate((G_for_H, zeroLeftCorner), axis=0)
    HLeft = HLeft.astype(np.double)
    SmallI = np.identity(2 * m) * 0.0001
    H = np.concatenate((HLeft, HRight), axis=1) + SmallI
    H = matrix(H)
    # print(np.linalg.eigvals(H).min())
    u_left = np.zeros(m)
    u_right = (1 / m) * np.ones(m)
    u = matrix(np.concatenate((u_left, u_right)))
    v_up = np.zeros((m, 1))
    v_down = np.ones((m, 1))

    v = matrix(np.concatenate((v_up, v_down)))
    A_topLeft = np.zeros((m, m))
    A_topRight = np.identity(m)
    A_downRight = np.identity(m)
    A_right = np.concatenate((A_topRight, A_downRight), axis=0)
    A_downLeft = np.array([np.double(trainy[i]) * G[i] for i in range(m)])
    A_left = np.concatenate((A_topLeft, A_downLeft), axis=0)
    A = np.concatenate((A_left, A_right), axis=1)
    A = matrix(A)
    sol = solvers.qp(H, u, -A, -v)
    return np.array(sol["x"][:m])


def retGramMatrix(trainX, testX, k):
    polyKernel = lambda vi, vj: (1 + np.double(np.inner(vi, vj))) ** k
    return np.array([[polyKernel(trainX[i], testX[j]) for i in range(len(trainX))] for j in range(len(testX))])


def plot_softsvm_prediction(X, y, lambda_, ks):
    # generate a grid of points for plotting the decision boundary
    x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))

    # create the list of patches for the legend
    patches = [plti.Patch(color='blue', label='-1'), plti.Patch(color='red', label='1')]

    # create a figure with subplots
    fig, axs = plt.subplots(1, len(ks), figsize=(12, 4))

    # loop over each value of k
    for ax, k in zip(axs, ks):
        # compute the alphas for the given value of k and lambda
        alphas = softsvmpoly(lambda_, k, X, y)

        # compute the predictions for the grid points using the alphas
        predictions = np.sign(np.dot(retGramMatrix(X, np.c_[x1.ravel(), x2.ravel()], k), alphas)).reshape(x1.shape)

        # plot the decision boundary
        ax.imshow(predictions, cmap=ListedColormap(['blue', 'red']), extent=[-10, 10, -10, 10])
        ax.set_title(f'Softsvm prediction on a 2D grid with lambda = {lambda_}, k = {k}')
        ax.legend(handles=patches)
    plt.show()


def simple_test():
    # load question 2 data
    data = np.load('ex2q4_data.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']
    # paintPoints(trainX, trainy)
    # S = np.array(list(zip(trainX, trainy)), dtype=object)
    # w, valueToParams = k_fold_non_kernel(softsvm, S, 5, [1, 10, 100])
    # print(f"The error on all the training data is: {err_soft(w, np.array(list(zip(testX, testy)), dtype=object))}")
    # for key, value in valueToParams.items():
    #     print(f"For the parameter l = {value} the error was: {key} ")
    plot_softsvm_prediction(trainX, trainy, 100, [3, 5, 8])


# S = []
    # for i in range(len(trainX)):
    #     S.append(np.array([trainX[i], trainy[i]], dtype=object))
    # w, k, valueToParams, values = k_fold(softsvmpoly, S, 5, [2, 5, 8], [1, 10, 100])
    # V = [np.array([testX[i], testy[i]], dtype=object) for i in range(len(testX))]
    # print(f"The error on the test data for the predicted seperator is: {err(w, V, k, trainX)}")
    # printErrors(valueToParams, values)
    # predicty = np.array([1 if pred(i)[0] == 1 else -1 for i in range(m)])
    # results.append(np.mean(predicty != _trainy))
    # tests to make sure the output is of the intended class and shape
    # assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    # assert w.shape[0] == m and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"
    # print(np.mean(predicty != _testY))


def paintPoints(trainX, trainY):
    colorsMap = {1: 'red', -1: 'blue'}
    fig, ax = plt.subplots(1, 1)
    x = [trainX[i][0] for i in range(len(trainX))]
    y = [trainX[i][1] for i in range(len(trainX))]
    colors = [colorsMap[trainY[i]] for i in range(len(trainY))]
    ax.scatter(x, y, c=colors)
    ax.set_title("Global")
    ax.set_xlabel("number tweets")
    ax.set_ylabel("mean engagement")
    fig = ax.get_figure()
    fig.savefig("4a.png")


def err(h_i, V, k, S_tag_x):
    _testY: np.array = np.array([y for _, y in V])
    _testX = np.array([x for x, _ in V])
    G = retGramMatrix(S_tag_x, _testX, k)
    return np.sum(np.sign(np.dot(G, h_i)) != _testY.reshape(_testY.shape[0], 1)) / _testY.shape[0]


def k_fold(A, S, number_of_folds, ks, lambdas):
    n = len(S)
    values = []
    valueToParams = {}
    for l in lambdas:
        for k in ks:
            epsilons = []
            for i in range(number_of_folds):
                start = (n // number_of_folds) * i
                end = start + (n // number_of_folds)
                V = S[start:end]
                part1 = np.array(S[:start])
                part2 = np.array(S[end:])
                if len(part1) == 0:
                    S_tag = part2
                elif len(part2) == 0:
                    S_tag = part1
                else:
                    S_tag = np.concatenate((part1, part2))
                S_tag_x = [x for x, _ in S_tag]
                S_tag_y = [y for _, y in S_tag]
                h_i = A(l, k, S_tag_x, S_tag_y)
                epsilon_i = err(h_i, V, k, S_tag_x)
                epsilons.append(epsilon_i)
            epsilon_parm = np.sum(epsilons) / len(epsilons)
            values.append((epsilon_parm, (l, k)))
            valueToParams[epsilon_parm] = (l, k)
    minParam = min(valueToParams.keys())
    l, k = valueToParams[minParam]
    return A(l, k, [x for x, _ in S], [y for _, y in S]), k, valueToParams, values


def k_fold_non_kernel(A, S, number_of_folds, lambdas):
    n = len(S)

    valueToParams = {}
    for l in lambdas:
        epsilons = []
        for i in range(number_of_folds):
            start = (n // number_of_folds) * i
            end = start + (n // number_of_folds)
            V = S[start:end]
            part1 = np.array(S[:start])
            part2 = np.array(S[end:])
            if len(part1) == 0:
                S_tag = part2
            elif len(part2) == 0:
                S_tag = part1
            else:
                S_tag = np.concatenate((part1, part2))
            S_tag_x = [x for x, _ in S_tag]
            S_tag_y = [y for _, y in S_tag]
            h_i = A(l, S_tag_x, S_tag_y)
            epsilon_i = err_soft(h_i, V)
            epsilons.append(epsilon_i)
        epsilon_parm = np.sum(epsilons) / len(epsilons)
        if epsilon_parm in valueToParams.keys():
            valueToParams[epsilon_parm + np.random.random() / 10000000] = l
        else:
            valueToParams[epsilon_parm] = l

    minParam = min(valueToParams.keys())
    l = valueToParams[minParam]
    return A(l, [x for x, _ in S], [y for _, y in S]), valueToParams


def err_soft(h_i, V):
    testX = np.array([x for x, _ in V])
    testy = np.array([y for _, y in V])
    predicty = np.array([1.0 if np.sign(testX[i] @ h_i)[0] == 1.0 else -1.0 for i in range(testX.shape[0])])
    return np.mean(predicty != testy)


def printErrors(valueToParams: {}, values):
    minParam = min(valueToParams.keys())
    l, k = valueToParams[minParam]
    data = []
    print(f"The hyper parameters with the lowest error are: lambda = {l}, k = {k} ")
    l = [l[0] for _, l in values]
    l = list(dict.fromkeys(l))
    k = [k[1] for _, k in values]
    k = list(dict.fromkeys(k))
    for key, value in values:
        # print(f"For lambda: {value[0]} and k: {value[1]} the average error is: {key}")
        data.append(key)
    data = np.array(data, dtype=object).reshape((3, 3))

    df = pd.DataFrame(data, index=l, columns=k)
    print(df)


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
