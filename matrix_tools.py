def get_split_matrix(raw_matrix, type_of_split='fst'):
    distance, azimuth = raw_matrix.shape

    if type_of_split == 'fst':
        distance_result, azimuth_result = 240, 256
    elif type_of_split == 'snd':
        distance_result, azimuth_result = 30, 128
    else:
        raise TypeError("Unexpected type of split")

    split_matrix = raw_matrix[0:distance - distance % distance_result, 0:azimuth - azimuth % azimuth_result].reshape(
        distance // distance_result, distance_result, -1, azimuth_result).swapaxes(1, 2).reshape(-1, distance_result,
                                                                                                 azimuth_result)
    return split_matrix


def get_null_coords_by_num(number_fst, number_snd=None):
    distance_indent_fst, azimuth_indent_fst = 240, 256
    count_in_row_fst = 5
    max_index_fst = 39

    if number_fst >= (max_index_fst + 1) or number_fst <= -1:
        raise IndexError("Unexpected index of chunk")

    distance_coordinate_fst = distance_indent_fst * (number_fst % count_in_row_fst)

    azimuth_coordinate_fst = azimuth_indent_fst * (number_fst // count_in_row_fst)

    if number_snd:

        distance_indent_snd, azimuth_indent_snd = 30, 128
        count_in_row_snd = 8
        max_index_snd = 15

        if number_snd >= (max_index_snd + 1) or number_snd <= -1:
            raise IndexError("Unexpected index of chunk")

        distance_coordinate_snd = distance_indent_snd * (number_snd % count_in_row_snd) + distance_coordinate_fst

        azimuth_coordinate_snd = azimuth_indent_snd * (number_snd // count_in_row_snd) + azimuth_coordinate_fst

        return distance_coordinate_snd, azimuth_coordinate_snd

    return distance_coordinate_fst, azimuth_coordinate_fst


def get_center_by_null_coords(distance_null_coord, azimuth_null_coord):
    distance_indent = 30
    azimuth_indent = 128

    distance_center_coord = distance_null_coord + distance_indent // 2
    azimuth_center_coord = azimuth_null_coord + azimuth_indent // 2

    return distance_center_coord, azimuth_center_coord


def get_kilometers_and_grades_by_coords(distance_coord, azimuth_coord):
    kilometers = int((distance_coord / 2048) * 360)
    grades = int((azimuth_coord / 1200) * 360)

    return kilometers, grades


# def kilometers_and_grades_to_coordinates(grades, kilometers):
# result_of_grades = int(grades * 2048 / 360)
# result_of_kilometers = int(kilometers * 1200 / 360)
# return result_of_grades, result_of_kilometers


if __name__ == "__main__":
    pass
