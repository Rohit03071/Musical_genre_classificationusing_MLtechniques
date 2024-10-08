import numpy as np
from random import random
from sklearn.model_selection import train_test_split
import tensorflow as tf



#array{[[0.1, 0.2], [0.2, 0.2]]}
#array{[[0.3], [0.4]]}

def generate_dataset(num_samples, test_size):

   x = np.array([[random() / 2 for _ in range(2)] for _ in range(num_samples)])  #array([[0.1, 0.2], [0.3, 0.4]])

   y = np.array([[i[0] + i[1]] for  i in x])   #array([[0.3], [0.7]])

   #whether network is been able to genralize the patterns it has learnt by training on a data set on training set
   #in order to do that we have to evaluate our model by proving the data, our model has never seen before

   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)

   return x_train, x_test, y_train, y_test


if __name__ == "__main__":
   
   x_train, x_test, y_train, y_test= generate_dataset(5000, 0.3)

   #build model
   model = tf.keras.Sequential([
      
      tf.keras.layers.Dense(5,  input_dim = 2, activation ="sigmoid"),
      tf.keras.layers.Dense(1,  activation ="sigmoid")

      ])
   
   optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)
   model.compile(optimizer = optimizer, loss="MSE")


   model.fit(x_train, y_train, epochs = 100, batch_size = 10)

   print("\n Model Evaluation")

   model.evaluate(x_test, y_test, verbose = 1)

   data = np.array([[0.1, 0.2], [0.2, 0.2]])

   predictions = model.predict(data)

   print("some predictions:")

   for d, p in zip(data, predictions):
      print("{} + {} = {}".format(d[0], d[1], p[0]))







   






