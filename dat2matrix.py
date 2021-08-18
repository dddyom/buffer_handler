import numpy as np


f = open("sources/SO_201207_153155.dat", "rb")
buffer = f.read()

matrix = []

for i in range(0, len(buffer)-1, 2):
    matrix.append(int(str(buffer[i + 1]) + str(buffer[i])))


matrix = np.array(matrix)
print(matrix.shape)

matrix = np.reshape(matrix, (2048, 1200))
print(matrix.shape)
print(matrix[0][5:10])

