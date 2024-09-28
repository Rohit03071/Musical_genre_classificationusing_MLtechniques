import numpy as np
from random import random


# save the activations and derivatives
#implement back propagation
#implement gradient descent 



#train our netwrok with dummy dataset


class MLP:
    def __init__(self, num_inputs = 2, hidden_layers = [5], num_outputs = 1):
        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers +[num_outputs]

        # initiate random weights

        self.weights = []

        for i in range(len(layers) - 1):

            w = np.random.rand(layers[i], layers[i+1])

            self.weights.append(w)


        activations = []

        for i in range(len(layers)):

            a = np.zeros(layers[i])

            activations.append(a)

        self.activations = activations


        derivatives = []

        for i in range(len(layers) - 1):

            d = np.zeros((layers[i], layers[i+1]))

            derivatives.append(d)

        self.derivatives = derivatives



    def _sigmoid(self, x):

        y = 1.0/(1 + np.exp(-x))

        return y


    def forward_propagate(self, inputs):

        activations = inputs

        self.activations[0] = inputs

        #iterate through network layers

        for i, w in enumerate(self.weights):

            #calaculate the net inputs
            net_inputs = np.dot(activations, w)

            #calaculate the activation

            activations = self._sigmoid(net_inputs)

            self.activations[i+1] = activations

            # a_3 = sigmoid(h_3)
            #h_3 = a_2 * w_2

        return activations
    
    def back_Propagate(self, error, verbose = False):

        #dE/dW_i = (y - a(i + 1))  *  s`(h_(i+1))  *  a_i

        #s`(h_(i+1)) = s(h_(i+1))(1 - s(h_(i+1)))

        #s(h_i+1) = a_(i+1)

        #dE/dW_(i -1)= (y - a(i + 1))  *  s`(h_(i+1))  *  W_i  *  s`(h_i)  *  a[i-1]

        for i in reversed(range (len(self.derivatives))):

            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)  # ndarray([0.1, 0.2])--> ndarray([[0.1 ,0.2]])

            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]  # ndarray([0.1, 0.2])--> ndarray([0.1], [0.2])

            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print(f"Derivatives for W{i}: {self.derivatives[i]}")

        return error

    


    def train(self, inputs, targets, epochs, learning_rate):
        for i in range(epochs):

            sum_error = 0

            for j, input in enumerate(inputs):

                target = targets[j]

                output = self.forward_propagate(input)

                error = target - output

                #back_propagation

                self.back_Propagate(error)

                #apply gradient descent

                self.gradient_descent(learning_rate)

                 #report the error
                sum_error += self._mse(target, output)

            print("Error: {} at epoch {}".format((sum_error / len(inputs)), i+ 1))

    
    
    def gradient_descent(self, learning_rate):

        for i in range(len(self.weights)):


            self.weights[i] = self.weights[i] + self.derivatives[i] * learning_rate


    def _sigmoid(self, x):
        """Sigmoid activation function
        Args:
            x (float): Value to be processed
        Returns:
            y (float): Output
        """

        y = 1.0 / (1 + np.exp(-x))
        return y
    

    def _sigmoid_derivative(self, x):

        return x * (1.0 - x)
    

    def _mse(self, target, output):

        return np.average((target - output)** 2)



if __name__ == "__main__":

    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])  #array([[0.1, 0.2], [0.3, 0.4]])

    targets = np.array([[i[0] + i[1]] for  i in inputs])   #array([[0.3], [0.7]])


    #reate an MLP

    mlp = MLP(2, [5], 1)


    mlp.train(inputs, targets, 50, 0.1)





