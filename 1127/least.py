# -*- coding: utf-8 -*-
import numpy as np


def main():
    # TODO: make f(x), phai, t
    X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    t = np.array([0.709745, 0.518762, 0.018762, -0.599272, -1.099272, -1.290255, -1.099272, -0.599272, 0.018762, 0.518762, 0.709745, 0.518762, 0.018762, -0.599272, -1.099272, -1.290255, -1.099272, -0.599272, 0.018762, 0.518762])

    m = 3

    # phai = np.empty(m + 1)
    print phai
    for x in X:
        col = np.array([])
        for i in range(0, m + 1):
            col = np.append(col, np.power(x, i))

        phai = np.row_stack((phai, col))

    print phai

    # w = np.linalg.inv(phai.T.dot(phai)).dot(phai.T).dot(t)

    print np.linalg.inv(phai.T.dot(phai)).dot(phai.T).dot(t)

if __name__ == '__main__':
    main()
