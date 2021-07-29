import numpy as np

from dev_config import PATH_TO_NPY_MATRIX as npy_path

def get_split_matrix(buffer_matrix, type_of_split='fst'):
    distance, azimuth = buffer_matrix.shape

    if type_of_split == 'fst':
        distance_result, azimuth_result = 240, 256
    elif type_of_split == 'snd':
        distance_result, azimuth_result = 30, 128
    else:
        raise TypeError("Unexpected type of split")

    result_split_matrix = buffer_matrix[0:distance - distance % distance_result, 0:azimuth - azimuth % azimuth_result].reshape(
        distance // distance_result, distance_result, -1, azimuth_result).swapaxes(1, 2).reshape(-1, distance_result, azimuth_result)
    return result_split_matrix


def get_null_coords_by_num(number, type_of_chunks='fst'):
    
    if type_of_chunks == 'fst':
        distance_indent, azimuth_indent = 240, 256
        count_in_row = 5
        max_index = 39
    elif type_of_chunks == 'snd':
        distance_indent, azimuth_indent = 30, 128
        count_in_row = 8
        max_index = 15
    else:
        raise TypeError("Unexpected type of chunk")
    
    if number >= (max_index + 1) or number <= -1:
        raise IndexError("Unexpected index of chunk")
    
    distance_coordinate = distance_indent * (number % count_in_row)
    
    azimuth_coordinate = azimuth_indent *(number // count_in_row)    
    
    #azimuth_coordinate = 0    
    #for i in range(-1, number, count_in_row):
        #if i == -1:
            #continue
        #azimuth_coordinate += azimuth_indent
        #print(i)
        
    return (distance_coordinate, azimuth_coordinate)

def get_center_by_null_coords(distance_null_coord, azimuth_null_coord):
    distance_indent = 30
    azimuth_indent = 128
    
    distance_center_coord = distance_null_coord + distance_indent // 2
    azimuth_center_coord = azimuth_null_coord + azimuth_indent // 2
    
    return (distance_center_coord, azimuth_center_coord)
    
def get_grades_and_kilometers_by_coords(distance_coord, azimuth_coord):
    
    kilometers = int((distance_coord / 2048) * 360)
    grades = int((azimuth_coord / 1200) * 360)
    
    return kilometers, grades

#def kilometers_and_grades_to_coordinates(grades, kilometers):
    #result_of_grades = int(grades * 2048 / 360)
    #result_of_kilometers = int(kilometers * 1200 / 360)
    #return result_of_grades, result_of_kilometers



def main():
    test = np.load(npy_path)
    #print(test.shape, '\n')
    split_matrix = get_split_matrix(test)
    #print(split_matrix.shape, '\n')
    #print(split_matrix[0])
    distance_null, azimuth_null = get_null_coords_by_num(number=39)
    print(get_grades_and_kilometers_by_coords(distance_null, azimuth_null))
    print(get_center_by_null_coords(distance_null, azimuth_null))


if __name__ == "__main__":
    main()
    pass



