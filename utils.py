import numpy as np


def matrix_normalize(matrix, row_normalize=False):
    if row_normalize:
        row_sums = matrix.sum(axis=1)
        return np.array([matrix[index, :] / row_sums[index] if row_sums[index] != 0 else [0] * row_sums.size for index in range(row_sums.size)])
    else:
        column_sums = matrix.sum(axis=0)
        return np.array([matrix[:, index] / column_sums[index] if column_sums[index] != 0 else [0]*column_sums.size for index in range(column_sums.size)]).T

