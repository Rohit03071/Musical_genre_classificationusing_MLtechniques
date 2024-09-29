import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


#load the data from dataset
DATASET_PATH = "data_10.json"

def load_dataset(dataset_path):
    with open(dataset_path, "r") as fp:

        data =json.load(fp)
        #convert lists into numpy arrays
        inputs = np.array(data["mfcc"])
        targets = np.array(data["labels"])

        return inputs, targets
    

def plot_history(history):
    fig, axs = plt.subplots(2, 1)

    #create the accuracy subplots

    axs[0].plot(history.history["accuracy"], label = "train_accuracy")
    axs[0].plot(history.history["val_accuracy"], label = "test_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label = "train_error")
    axs[1].plot(history.history["val_loss"], label = "test_error")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Error")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


if __name__ == "__main__":
    inputs, targets = load_dataset(DATASET_PATH)
    
    #split the data into train and test sets

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)

    
    #build network architecture

    model = keras.Sequential([

        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidde layer

        keras.layers.Dense(512, activation = "relu", kernel_regularizer=keras.regularizers.l1(0.0001)),

        keras.layers.Dense(256, activation = "relu", kernel_regularizer=keras.regularizers.l1(0.0001)),

        keras.layers.Dense(32, activation = "relu", kernel_regularizer=keras.regularizers.l1(0.0001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(10, activation="softmax")

    ])

    #compile the network


    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer = optimizer, 
                  loss= "sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()


    #tran the network

    history = model.fit(inputs_train, targets_train, validation_data=(inputs_test, targets_test),
                                                            epochs = 50, 
                                                            batch_size=32)
    plot_history(history)
