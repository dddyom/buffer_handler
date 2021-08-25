import os
import time

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
    
    load_fst_neural_start = time.perf_counter()
    first_trained_neural = Neural(
        240, 256, checkpoint_path + "/first_trained.ckpt")
    print("First neural boot time: %.2f" % (time.perf_counter() - load_fst_neural_start))

    load_snd_neural_start = time.perf_counter()
    second_trained_neural = Neural(
        30, 128, checkpoint_path + "/second_trained.ckpt")
    print("Second neural boot time: %.2f" % (time.perf_counter() - load_snd_neural_start))


    loading_dat_file_list_start = time.perf_counter()
    all_files = os.listdir(dat_path)
    all_dat = filter(lambda x: x.endswith('.dat'), all_files)
    print("Loading dat file list time: %.2f" % (time.perf_counter() - loading_dat_file_list_start))
    for i in all_dat:
        convert_dat_to_numpy_start = time.perf_counter()
        np_matrix = dat2nparr(dat_path + "/" + i)
        print("Convert dat file to numpy for %(1)s time: %(2).2f" % {"1": i, "2": (time.perf_counter() - convert_dat_to_numpy_start)})

        split_fst_start = time.perf_counter()
        first_split_matrix = matrix_tools.get_split_matrix(np_matrix)
        print("First split time for %(1)s is: %(2).2f" % {"1": i, "2": (time.perf_counter() - split_fst_start)})
        predict_fst_start = time.perf_counter()
        first_trained_neural.predict(first_split_matrix)
        print("First predict time for %(1)s is: %(2).2f" % {"1": i, "2": (time.perf_counter() - predict_fst_start)})
        # print(first_trained_neural.sug_targets_ind)
        second_split_and_predict_start = time.perf_counter()
        sug_targets_before_first = first_trained_neural.sug_targets
        for j in range(len(sug_targets_before_first)):
            second_split_matrix = matrix_tools.get_split_matrix(sug_targets_before_first[j], type_of_split="snd")
            second_trained_neural.predict(second_split_matrix)

        print("Second split and predict time for %(1)s is : %(2).2f" % {"1": i, "2": (time.perf_counter() - second_split_and_predict_start)})
        # print(second_trained_neural.sug_targets_ind)
        convert_coordinates_start = time.perf_counter()
        for k in second_trained_neural.sug_targets_ind:
            # print(first_trained_neural.sug_targets_ind[i], i)
            distance_null_coord, azimuth_null_coord = matrix_tools.get_null_coords_by_num(
                first_trained_neural.sug_targets_ind[k], k)
            distance_center_coord, azimuth_center_coord = matrix_tools.get_center_by_null_coords(distance_null_coord,
                                                                                                 azimuth_null_coord)
            kilometers, grades = matrix_tools.get_kilometers_and_grades_by_coords(distance_center_coord,
                                                                                  azimuth_center_coord)
            print(kilometers, grades)
        print("Convert coordinates time for %(1)s is %(2).2f" % {"1": i, "2": (time.perf_counter() - convert_coordinates_start)})

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
