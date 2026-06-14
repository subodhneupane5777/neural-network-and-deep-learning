import numpy as np

def nth_smallest(data, top, bottom, left, right, n):
    # extract the subarray using slicing
    sub_matrix = data[top:bottom + 1, left:right + 1]

    # flatten the subarray into a 1D array
    flattened = sub_matrix.flatten()

    # sort the flattened array
    sorted_values = np.sort(flattened)

    # return the n-th smallest value (1-based indexing)
    return sorted_values[n - 1]

# Test code
if __name__ == "__main__":
    a = np.array([[0.8147, 0.0975, 0.1576, 0.1419, 0.6557, 0.7577],
                  [0.9058, 0.2785, 0.9706, 0.4212, 0.4212, 0.7431],
                  [0.1270, 0.5469, 0.9572, 0.9157, 0.8491, 0.3922],
                  [0.9134, 0.9575, 0.4854, 0.7922, 0.9340, 0.6555],
                  [0.6324, 0.9649, 0.8003, 0.9595, 0.6787, 0.1712]])

    print(nth_smallest(a, 1, 2, 3, 4, 1))
    print(nth_smallest(a, 1, 2, 3, 4, 2))
    print(nth_smallest(a, 1, 2, 3, 4, 3))
    print(nth_smallest(a, 1, 2, 3, 4, 4))
