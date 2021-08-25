import os

import matrix_tools
from buffer_handler_init import DAT_PATH, CHECKPOINT_PATH
from dat2matrix import dat2nparr
from Neural import Neural

try:
    dat_path = os.path.normpath(DAT_PATH)
except Exception:
    ValueError(f"Unexpected path: {DAT_PATH}")

try:
    checkpoint_path = os.path.normpath(CHECKPOINT_PATH)
except Exception:
    ValueError(f"Unexpected path{CHECKPOINT_PATH}")


# def first_handle(trained_model, np_matrix):
#     first_split_matrix = matrix_tools.get_split_matrix(np_matrix)
#     first_np_targets, first_targets_i = load_fst.first_predict(
#         first_split_matrix)
#     return first_np_targets, first_targets_i


# def second_handle(first_np_targets, first_targets_i):
#
#     second_targets_i = load_snd.second_predict(
#         first_np_targets, first_targets_i)
#     return second_targets_i


def main():
    first_trained_neural = Neural(
        240, 256, checkpoint_path + "/first_trained.ckpt")
    second_trained_neural = Neural(
        30, 128, checkpoint_path + "/second_trained.ckpt")

    all_files = os.listdir(dat_path)
    all_dat = filter(lambda x: x.endswith('.dat'), all_files)
    for i in all_dat:
        np_matrix = dat2nparr(dat_path + "/" + i)
        first_split_matrix = matrix_tools.get_split_matrix(np_matrix)
        first_trained_neural.predict(first_split_matrix)
        # print(first_trained_neural.sug_targets_ind)
        for i in range(len(first_split_matrix)):
            second_split_matrix = matrix_tools.get_split_matrix(first_split_matrix[i], type_of_split="snd")
            second_trained_neural.predict(second_split_matrix)
        # print(second_trained_neural.sug_targets_ind)
        for i in second_trained_neural.sug_targets_ind:
            # print(first_trained_neural.sug_targets_ind[i], i)
            distance_null_coord, azimuth_null_coord = matrix_tools.get_null_coords_by_num(
                first_trained_neural.sug_targets_ind[i], i)
            distance_center_coord, azimuth_center_coord = matrix_tools.get_center_by_null_coords(distance_null_coord,
                                                                                                 azimuth_null_coord)
            kilometers, grades = matrix_tools.get_kilometers_and_grades_by_coords(distance_center_coord,
                                                                                  azimuth_center_coord)
            print(kilometers, grades)

    #     first_np_targets, first_targets_i = first_handle(matrix)
    #     second_targets_i = second_handle(first_np_targets, first_targets_i)
    #     for i in second_targets_i:
    #         fst_i = int(np.where(values_i_fst == i)[0])
    #         snd_i = i

    #     distance_null_coord, azimuth_null_coord = get_null_coords_by_num(
    #         fst_i, snd_i)
    #     print(distance_null_coord, azimuth_null_coord)

    #     distance_coord, azimuth_coord = get_center_by_null_coords(
    #         distance_null_coord=distance_null_coord, azimuth_null_coord=azimuth_null_coord)
    #     print(distance_coord, azimuth_coord)

    #     kilometers, grades = get_grades_and_kilometers_by_coords(
    #         distance_coord=distance_coord, azimuth_coord=azimuth_coord)
    #     print(kilometers, grades)
    # print(f"for {i} shape is {matrix.shape}")


if __name__ == "__main__":
    main()
