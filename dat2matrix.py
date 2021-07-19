import numpy as np
import struct


matrix = []
with open("sources/SO_201127_151029.dat", "rb") as f:
    for i in f:
        print(struct.unpack('f', i))
        #matrix.append(np.array(list(i)))
        #print(np.array(list(i)).shape)

#matrix = np.array(matrix)
#print(matrix)
#print(matrix.shape)
#np.save(matrix, '1.npy')
