from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np


class Neural:
    def __init__(self, width, length, checkpoint_path):
        self.model = keras.Sequential([
            Flatten(input_shape=(width, length, 1)),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.load_weights(checkpoint_path)

        self.sug_targets = None
        self.sug_targets_ind = None

    def get_model(self):
        return self.model

    def predict(self, split_matrix):
        suggested_targets = []
        values_i = []

        for i in range(len(split_matrix)):
            temp = np.expand_dims(split_matrix[i], axis=0)
            img_expended = np.expand_dims(temp, axis=3)

            prediction = self.model.predict(img_expended)[0]

            if prediction[1] > prediction[0]:
                # target
                suggested_targets.append(split_matrix[i])
                values_i.append(i)

        self.sug_targets = np.array(suggested_targets)
        self.sug_targets_ind = np.array(values_i)


if __name__ == "__main__":
    pass
    # import tensorflow as tf
    # from tensorflow.keras.models import load_model
    # import matplotlib.pyplot as plt
    # from tensorflow.keras.utils import to_categorical
    # import os

    # def load_data(name):
    # """Возвращает массиы с данными из npz по имени"""
    # data = np.load(name)
    # return data['x_train'], data['x_test'], data['y_train'], data['y_test']

    # def prepare_dataset(name_of_dataset):
    # x_train, x_test, y_train, y_test = load_data(name_of_dataset)
    #
    # y_train_in = to_categorical(y_train, 2)
    # x_train_in = x_train / 255
    # y_test_in = to_categorical(y_test, 2)
    # x_test_in = x_test / 255
    #
    # x_train_in = np.expand_dims(x_train, axis=3)
    # x_test_in = np.expand_dims(x_test, axis=3)
