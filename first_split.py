import numpy as np

from dev_config import PATH_TO_NPY_MATRIX as npy_path

def get_split_matrix(buffer_matrix, type_of_split='fst'):
    width, length = buffer_matrix.shape

    if type_of_split == 'fst':
        width_result, length_result = 240, 256
    elif type_of_split == 'snd':
        width_result, length_result = 30, 128
    else:
        raise TypeError("Unexpected type of split")

    result_split_matrix = buffer_matrix[0:width - width % width_result, 0:length - length % length_result].reshape(
        width // width_result, width_result, -1, length_result).swapaxes(1, 2).reshape(-1, width_result, length_result)
    return result_split_matrix


def main():
    test = np.load(npy_path)
    print(test.shape, '\n')
    split_matrix = get_split_matrix(test)
    print(split_matrix.shape, '\n')
    print(split_matrix[0])


if __name__ == "__main__":
    main()
    pass
