import numpy as np
import scipy as sp

class AIMath():

    def Neuron(self, input_dim, weights, bias):
        self.input_dim = input_dim
        self.weights = set(weight)
        self.bias = bias
        #let's give the number of inputs to the neuron
        self.input_dim = int(input("Give an input: "))

        weights = [1,2,3]

        bias = 0.3

        def activation(x, y, z):
            #performs the operation
            return ((x * weights[0]) + (y * weights[1]) + (z * weights[2])) * bias
            
        print(activation(num1, num2, num3))
        cycles = int(input("Enter the number of cycles you wish: "))

        for i in range(cycles):
            print(activation(num1, num2, num3))
            num1 = num1 * bias
            num2 = num2 * bias
            num3 = num3 * bias
            bias = cycles / bias


    def HopField(self, input_dim, vectors):
        self.input_dim = input_dim
        self.vectors = np.array(vectors)
        