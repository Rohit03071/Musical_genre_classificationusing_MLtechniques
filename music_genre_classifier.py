import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras


#load the data from dataset
DATASET_PATH = "data.json"

def load_dataset(dataset_path):
    with open(dataset_path, "r") as fp:

        data =json.load(fp)
        #convert lists into numpy arrays
        inputs = np.array(data["mfcc"])
        targets = np.array(data["labels"])

        return inputs, targets

if __name__ == "__main__":
    inputs, targets = load_dataset(DATASET_PATH)
    
    #split the data into train and test sets

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    
    #build network architecture

    model = keras.Sequential([

        keras.layers.Flatten(input_shape=(input.shape[1], input.shape[2])),

        # 1st hidde layer

        keras.layers.Dense(512, activation = "relu"),

        keras.layers.Dense(256, activation = "relu"),

        keras.layers.Dense(32, activation = "relu"),

        keras.layers.Dense(10, activations="softmax")

    ])

    #compile the network


    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer = optimizer, loss= "sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    #tran the network

    model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test),
                                                            epochs = 50, 
                                                            batch_size=32)
    





