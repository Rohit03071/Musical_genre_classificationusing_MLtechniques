import json
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split



DATA_PATH = "D:/MachineLearning/data_10.json"

def load_data(data_path):

    with open(data_path,"r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y

def prepare_dataset(test_size, validation_size):
    #we will load the data
    X, y = load_data(DATA_PATH)

    #will be creating train/test split to define how much data will be trained and how much data will be reserved for test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    #will be creating train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    #3d array -> (130, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, y_train, y_validation, X_test, y_test

def build_model(input_shape):

    #starting to create the main model
    model = keras.Sequential()

    #adding convolution layers

    model.add(keras.layers.Conv2D(32, (3,3), activation ="relu", input_shape=input_shape))

    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3,3), activation ="relu", input_shape=input_shape))

    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding="same"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (2,2), activation ="relu", input_shape=input_shape))

    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding="same"))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation="relu"))

    model.add(keras.layers.Dropout(0.3))

    
    #flatten the output and feed into the dense layer later
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X) #X ==> (1, 130, 13, 1)

    # prediction = [[0.1, 0.2, 0.3, ...]]

    #we can't really predict one sample based on the 2d array we are geeting
    # we have to find the maximum value from the array to finally print the predicted value
    # so we are gonna extract index from the array with a maximum value

    prediction_index = np.argmax(prediction, axis = 1) #we are gonna get somthing like this [4]

    print("the expected label index value is: {}, Predicted Index value is: {}".format(y, prediction_index))










if __name__ == "__main__":
    #creating train, validation and test sets one by one

    X_train, X_validation, y_train, y_validation, X_test, y_test = prepare_dataset(0.25, 0.2)

    #building the Convolution N N 

    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    #compile the network

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer = optimizer,
                   loss="sparse_categorical_crossentropy",
                   metrics = ['accuracy'])

    #train the network

    model.fit(X_train, y_train, validation_data = (X_validation, y_validation), batch_size = 32, epochs = 30)


    #evaluate the cnn on test set

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose = 1)

    print("Accuracy on test set is: {}".format(test_accuracy))

    #we will be doing prediction for our trained model

    X = X_test[100]
    y = y_test[100]

    predict(model, X, y)

