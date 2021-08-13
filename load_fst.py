import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np


def load_data(name):
    """Возвращает массиы с данными из npz по имени"""
    data = np.load(name)
    return data['x_train'], data['x_test'], data['y_train'], data['y_test']
    

#def prepare_dataset(name_of_dataset):
    #x_train, x_test, y_train, y_test = load_data(name_of_dataset)
#
    #y_train_in = to_categorical(y_train, 2)
    #x_train_in = x_train / 255
    #y_test_in = to_categorical(y_test, 2)
    #x_test_in = x_test / 255
#
    #x_train_in = np.expand_dims(x_train, axis=3)
    #x_test_in = np.expand_dims(x_test, axis=3)


def create_model():
    model = keras.Sequential([
        Flatten(input_shape=(240, 256, 1)),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])    
    return model

#def show():
        #for i in range(len(x_test)):
        #qwe = np.expand_dims(x_test[i], axis=0)
        #img_expended = np.expand_dims(qwe, axis=3)
        #prediction = model.predict(img_expended)[0]
        #if prediction[1] > prediction[0]:
            #title = 'цель'
    #
            #if y_test[i] == 1:
                #TP += 1 
            #else:
                #FN += 1
                #error_log.append(str(i) + '_defined_as_target')
        #else:
            #title = 'помехи'
            #if y_test[i] == 1:
                #FP += 1 
                #error_log.append(str(i) + '_defined_as_stray')
            #else:
                #TN += 1
    #plt.figure()
    #plt.imshow(x_test[i])
    #plt.title(f'{title}')
#def main():
    #model = create_model()
    #model.load_weights(checkpoint_path)
    #prepare_dataset("../neurals/dataset_256_240_main.npz")
    #
    #x_train, x_test, y_train, y_test = load_data("../neurals/dataset_256_240_main.npz")
#
    #y_train_in = to_categorical(y_train, 2)
    #x_train_in = x_train / 255
    #y_test_in = to_categorical(y_test, 2)
    #x_test_in = x_test / 255
#
    #x_train_in = np.expand_dims(x_train, axis=3)
    #x_test_in = np.expand_dims(x_test, axis=3)
    #
    #for i in range(len(x_test)):
        #qwe = np.expand_dims(x_test[i], axis=0)
        #img_expended = np.expand_dims(qwe, axis=3)
        #prediction = model.predict(img_expended)[0]
        #if prediction[1] > prediction[0]:
            #title = 'цель'
    #
            #if y_test[i] == 1:
                #TP += 1 
            #else:
                #FN += 1
                #error_log.append(str(i) + '_defined_as_target')
        #else:
            #title = 'помехи'
            #if y_test[i] == 1:
                #FP += 1 
                #error_log.append(str(i) + '_defined_as_stray')
            #else:
                #TN += 1
    #
    #plt.figure()
    #plt.imshow(x_test[i])
    #plt.title(f'{title}')


if __name__ == "__main__":
    checkpoint_path = "..neurals/trained_fst/cp.ckpt"  
    
    model = create_model()    
    model.load_weights(checkpoint_path)
    #prepare_dataset("../neurals/dataset_256_240_main.npz")
    
    x_train, x_test, y_train, y_test = load_data("../neurals/datasets/dataset_256_240_main.npz")

    y_train_in = to_categorical(y_train, 2)
    x_train_in = x_train / 255
    y_test_in = to_categorical(y_test, 2)
    x_test_in = x_test / 255

    x_train_in = np.expand_dims(x_train, axis=3)
    x_test_in = np.expand_dims(x_test, axis=3)
    
    suggested_targets = []
    values_i = []
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(x_test)):
        qwe = np.expand_dims(x_test[i], axis=0)
        img_expended = np.expand_dims(qwe, axis=3)
        prediction = model.predict(img_expended)[0]
        if prediction[1] > prediction[0]:
            title = 'цель'
            suggested_targets.append(x_test[i])
            values_i.append(i)        
            if y_test[i] == 1:
                TP += 1 
            else:
                FN += 1
                #error_log.append(str(i) + '_defined_as_target')
        else:
            title = 'помехи'
            if y_test[i] == 1:
                FP += 1 
                #error_log.append(str(i) + '_defined_as_stray')
            else:
                TN += 1
    
        #plt.figure()
        #plt.imshow(x_test[i])
        #plt.title(f'{title}')
        #plt.show()
    np.save("sources/processed/chunks/chunks_after_fst.npy", np.array(suggested_targets))
    np.save("sources/processed/i_values/i_fst.npy", np.array(values_i))