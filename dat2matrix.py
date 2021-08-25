import numpy as np
import struct


def dat2nparr(dat_file_dir):
    buffer = []
    dat_file = open(dat_file_dir, "rb").read()

    for i in range(0, len(dat_file), 2):
        buffer.append(struct.unpack("<H", dat_file[i: i + 2]))

    matrix = np.reshape(np.array(buffer), (2048, 1200))

    return matrix


if __name__ == "__main__":
    pass
    # import time
    # start_load_matrix = time.perf_counter()
    # matrix = dat2nparr("sources/SO_201207_153155.dat")
    # finish_load_matrix = time.perf_counter()
    # print(matrix.shape)
    # print(matrix[0][5:10])
    # print(finish_load_matrix - start_load_matrix)
