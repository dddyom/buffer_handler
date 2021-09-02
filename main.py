import os
import time
# from datetime import datetime
import logging

import matrix_tools
from buffer_handler_init import DAT_PATH, CHECKPOINT_PATH, VERSION
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

file_log = logging.FileHandler('buffer_handler_log')
console_out = logging.StreamHandler()

logging.basicConfig(handlers=(file_log, console_out), 
                    format='[%(asctime)s | %(levelname)s]: %(message)s', 
                    datefmt='%m.%d.%Y %H:%M:%S',
                    level=logging.INFO)


def main():
    logging.info(f"buffer_handler {VERSION} start")

    # log = open("buffer_handler_log", 'a')
    # log.write(f"\n\nLaunch the program: {datetime.now()}\n")
    # all_time = time.perf_counter()
    # load_fst_neural_start = time.perf_counter()
    logging.info("first trained neural boot")
    first_trained_neural = Neural(
        240, 256, checkpoint_path + "/first_trained.ckpt")
    # log.write("First neural boot time: %.2f \n" % (time.perf_counter() - load_fst_neural_start))

    # load_snd_neural_start = time.perf_counter()
    logging.info("second trained neural boot")
    second_trained_neural = Neural(
        30, 128, checkpoint_path + "/second_trained.ckpt")
    # log.write("Second neural boot time: %.2f \n" % (time.perf_counter() - load_snd_neural_start))

    # loading_dat_file_list_start = time.perf_counter()
    logging.info("Load dat file list")
    all_files = os.listdir(dat_path)
    all_dat = filter(lambda x: x.endswith('.dat'), all_files)
    # log.write("Loading dat file list time: %.2f \n" % (time.perf_counter() - loading_dat_file_list_start))
    for i in all_dat:
        logging.info(f"convert {i} to numpy array")

        # convert_dat_to_numpy_start = time.perf_counter()
        np_matrix = dat2nparr(dat_path + "/" + i)
        # log.write("Convert dat file to numpy for %(1)s time: %(2).2f \n" % {"1": i, "2": (
                    # time.perf_counter() - convert_dat_to_numpy_start)})

        # split_fst_start = time.perf_counter()
        logging.info(f"Split {i} numpy array")

        first_split_matrix = matrix_tools.get_split_matrix(np_matrix)
        # log.write("First split time for %(1)s is: %(2).2f \n" % {"1": i, "2": (time.perf_counter() - split_fst_start)})
        # predict_fst_start = time.perf_counter()
        logging.info("First neural predict")
        first_trained_neural.predict(first_split_matrix)
        # log.write(
            # "First predict time for %(1)s is: %(2).2f \n" % {"1": i, "2": (time.perf_counter() - predict_fst_start)})
        # print(first_trained_neural.sug_targets_ind)
        # second_split_and_predict_start = time.perf_counter()
        sug_targets_before_first = first_trained_neural.sug_targets

        logging.info(f"Split {i} for suggested targets before first predict")
        logging.info("Second neural predict")
        for j in range(len(sug_targets_before_first)):
            second_split_matrix = matrix_tools.get_split_matrix(sug_targets_before_first[j], type_of_split="snd")
            second_trained_neural.predict(second_split_matrix)

        # log.write("Second split and predict time for %(1)s is : %(2).2f \n" % {"1": i, "2": (
                    # time.perf_counter() - second_split_and_predict_start)})
        # print(second_trained_neural.sug_targets_ind)
        # convert_coordinates_start = time.perf_counter()
        logging.info("Convert coordinates")
        for k in second_trained_neural.sug_targets_ind:
            # print(first_trained_neural.sug_targets_ind[i], i)
            distance_null_coord, azimuth_null_coord = matrix_tools.get_null_coords_by_num(
                first_trained_neural.sug_targets_ind[k], k)
            distance_center_coord, azimuth_center_coord = matrix_tools.get_center_by_null_coords(distance_null_coord,
                                                                                                 azimuth_null_coord)
            kilometers, grades = matrix_tools.get_kilometers_and_grades_by_coords(distance_center_coord,
                                                                                  azimuth_center_coord)
            logging.info(f"Targets for {i} was found")
            logging.info(f"Coordinates: kilometers --> {kilometers}; grades --> {grades}")
            # print(kilometers, grades)
            # log.write(f"Found target : {kilometers, grades}\n")
        # log.write("Convert coordinates time for %(1)s is %(2).2f \n" % {"1": i, "2": (
                    # time.perf_counter() - convert_coordinates_start)})
    # log.write("All time: %.2f \n" % (time.perf_counter() - all_time))
        logging.info(f"Finish search for {i}")
    logging.info("Finish program")

    # log.close()


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")

    main()
