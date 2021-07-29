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


def get_null_coords_by_num(number=3, type_of_chunks='fst'):
    
    if type_of_chunks == 'fst':
        width_indent, length_indent = 240, 256
        count_in_row = 5
    elif type_of_chunks == 'snd':
        width_indent, length_indent = 30, 128
        count_in_row = 8
    
    width_axis = 0    
    for i in range(0, number, count_in_row):
        width_axis += width_indent
    
    length_axis = length_indent * (number % count_in_row)
    return (width_axis, length_axis)

def main():
    test = np.load(npy_path)
    print(test.shape, '\n')
    split_matrix = get_split_matrix(test)
    print(split_matrix.shape, '\n')
    print(split_matrix[0])
    print(get_null_coords_by_num(number=39))


if __name__ == "__main__":
    main()
    pass
