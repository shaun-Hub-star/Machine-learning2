import numpy as np
from cvxopt import solvers, matrix, spmatrix, spdiag, sparse
import matplotlib.pyplot as plt


# todo: complete the following functions, you may add auxiliary functions or define class to help you
def softsvmpoly(l: float, k: int, trainX: np.array, trainy: np.array):
    """

    param l: the parameter lambda of the soft SVM algorithm
    param k: the bandwidth parameter sigma of the RBF kernel.
    param trainX: numpy array of size (m, d) containing the training sample
    param trainy: numpy array of size (m, 1) containing the labels of the training sample
    return: numpy array of size (m, 1) which describes the coefficients found by the algorithm
    """
    G = retGramMatrix(trainX, k)
    G_for_H = 2*l*G
    m = len(trainy)
    d = len(trainX[0])

    HRight = np.zeros((d + m, m))
    zeroLeftCorner = np.zeros((m, d))
    HLeft = np.concatenate((G_for_H, zeroLeftCorner), axis=0)
    H = matrix(np.concatenate((HLeft, HRight), axis=1))
    u_left = np.zeros(d)
    u_right = (1 / m) * np.ones(m)
    u = matrix(np.concatenate((u_left, u_right)))
    v_up = np.zeros((m, 1))
    v_down = np.ones((m, 1))

    v = matrix(np.concatenate((v_up, v_down)))
    A_topLeft = np.zeros((m, d))
    A_topRight = np.identity(m)
    A_downRight = np.identity(m)
    A_right = np.concatenate((A_topRight, A_downRight), axis=0)
    A_downLeft = np.array(trainy[i]*G[i] for i in range(m))
    A_left = np.concatenate((A_topLeft, A_downLeft), axis=0)
    A = np.concatenate((A_left, A_right), axis=1)
    A = matrix(A)
    sol = solvers.qp(H, u, -A, -v)
    return np.array(sol["x"][:d])

def retGramMatrix(trainX, k):
    polyKernel = lambda vi, vj: (1 + np.inner(vi, vj)) ** k
    return np.array([polyKernel(trainX[i], trainX[j]) for i in range(len(trainX))] for j in range(len(trainX)))


def simple_test():
    # load question 2 data
    data = np.load('EX2q2_mnist.npz')
    trainX = data['Xtrain']
    testX = data['Xtest']
    trainy = data['Ytrain']
    testy = data['Ytest']

    m = 100

    # Get a random m training examples from the training set
    indices = np.random.permutation(trainX.shape[0])
    _trainX = trainX[indices[:m]]
    _trainy = trainy[indices[:m]]

    # run the softsvmpoly algorithm
    w = softsvmpoly(10, 5, _trainX, _trainy)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(w, np.ndarray), "The output of the function softsvmbf should be a numpy array"
    assert w.shape[0] == 1 and w.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    simple_test()

    # here you may add any code that uses the above functions to solve question 4
