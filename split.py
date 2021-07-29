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


def get_null_coords_by_num(number, type_of_chunks='fst'):
    
    if type_of_chunks == 'fst':
        width_indent, length_indent = 240, 256
        count_in_row = 5
        max_index = 39
    elif type_of_chunks == 'snd':
        width_indent, length_indent = 30, 128
        count_in_row = 8
        max_index = 15
    else:
        raise TypeError("Unexpected type of chunk")
    
    if number >= (max_index + 1) or number <= -1:
        raise IndexError("Unexpected index of chunk")
    
    width_coordinate = width_indent * (number % count_in_row)
    
    length_coordinate = length_indent *(number // count_in_row)    
    
    #length_coordinate = 0    
    #for i in range(-1, number, count_in_row):
        #if i == -1:
            #continue
        #length_coordinate += length_indent
        #print(i)
        
    return (width_coordinate, length_coordinate)

def get_center_by_null_coords(width_null_coord, length_null_coord):
    width_indent = 30
    length_indent = 128
    
    width_center_coord = width_null_coord + width_indent // 2
    length_center_coord = length_null_coord + length_indent // 2
    
    return (width_center_coord, length_center_coord)
    

def main():
    test = np.load(npy_path)
    #print(test.shape, '\n')
    split_matrix = get_split_matrix(test)
    #print(split_matrix.shape, '\n')
    #print(split_matrix[0])
    width_null, length_null = get_null_coords_by_num(number=39)
    #print(get_center_by_null_coords(width_null, length_null))


if __name__ == "__main__":
    main()
    pass
