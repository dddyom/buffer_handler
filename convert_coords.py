import numpy as np

from matrix_tools import get_null_coords_by_num, get_center_by_null_coords, get_grades_and_kilometers_by_coords

values_i_fst = np.load("sources/processed/i_values/i_fst.npy")
values_i_snd = np.load("sources/processed/i_values/i_snd.npy")

for i in values_i_snd:
    fst_i = int(np.where(values_i_fst==i)[0])
    snd_i = i

distance_null_coord, azimuth_null_coord = get_null_coords_by_num(fst_i, snd_i)
print(distance_null_coord, azimuth_null_coord)

distance_coord, azimuth_coord = get_center_by_null_coords(distance_null_coord=distance_null_coord, azimuth_null_coord=azimuth_null_coord)
print(distance_coord, azimuth_coord)

kilometers, grades = get_grades_and_kilometers_by_coords(distance_coord=distance_coord, azimuth_coord=azimuth_coord)
print(kilometers, grades)