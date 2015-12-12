# -*- coding: utf-8 -*-
import numpy as np


def main():
    # TODO: make f(x), phai, t
    X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    t = np.array(
        [0.709745, 0.518762, 0.018762, -0.599272, -1.099272, -1.290255, -1.099272, -0.599272, 0.018762, 0.518762,
         0.709745, 0.518762, 0.018762, -0.599272, -1.099272, -1.290255, -1.099272, -0.599272, 0.018762, 0.518762])

    for m in range(0, 21):
        w = genW(X, t, m)
        rmse = calRMSE(X, t, w)
        print "m:", m, "RMSE:", rmse, "w:", w


def genW(X, t, m=1):
    phai = np.zeros((len(X), m + 1))

    i = 0
    for x in X:
        for j in range(0, m + 1):
            phai[i, j] = np.power(x, j)
        i += 1

    # w = np.linalg.inv(phai.T.dot(phai)).dot(phai.T).dot(t)
    return np.linalg.inv(phai.T.dot(phai)).dot(phai.T).dot(t)

#
def calRMSE(X, t, w):
    sum = 0
    n = len(X)
    for i in range(0, n):
        sum += np.power(t[i] - calHypo(X[i], w), 2)

    return np.sqrt(sum / n)


# hypo = h0 + h1*x +h2*x^2 ..... + hM * x^M
def calHypo(x, w):
    sum = 0
    for j in range(0, len(w)):
        sum += w[j] * np.power(x, j)

    return sum

if __name__ == '__main__':
    main()
