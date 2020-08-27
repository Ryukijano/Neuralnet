import numpy as np

class FFNN():

    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        #adding 1 as it willl be our bias
        self.input_size = input_size + 1
        self.hidden_size = hidden_size + 1
        self.output_size = output_size

        self.o_error = 0
        self.o_delta = 0
        self.z1 = 0
        self.z2 = 0
        self.z3 = 0
        self.z2_error = 0

        # The whole weight matrix, from the inputs till the hidden layer
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        # The final set of weights from the hidden layer till the output layer
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(self,s):
        #Activation
        return 1 / (1 + np.exp(-s))

    def sigmoid_prime(self, s):
        #derivative of sigmoid
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def forward_pass(self, X):
        # Forward propagation through our network
        X['bias'] = 1
        self.z1 = np.dot(X, self.w1) # dot product of weight and input
        self.z2 = self.sigmoid(self.z1) # sigmoid
        self.z3 = np.dot(self.z2, self.w2) #dot product of weights and input
        o = self.sigmoid(self.z3)
        return o

    def prediction(self, X):
        return forward_pass(self, X)
    
    def backward_pass(self, X, y, output, step):
        X['bias'] = 1 #adding one to the inputs to include the bias in the weight
        self.o_error = y - output # ouput error
        self.o_delta = self.o_error * self.sigmoid_prime(output) * step # applying derivative of sigmoid error

        self.z2_error = self.o_delta.dot(
            self.w2.T) # z2 error: how much our hidden layer weights contribuuted to output error
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2) * step #applying derivatives of sigmoid to z2 error

        self.w1 = X.T.dot(self.z2_delta) #adjusting first set of weights
        self.w2 = self.z2.T.dot(self.o_error) #adjusting second set of weights

    def fit(self, X, y, epochs=10, step=0.05):
        for epoch in range(epochs):
            X['bias'] = 1 #Addind 1 in input to include the bias in the weight
            output = self.forward_pass(X)
            self.backward_pass(X, y, output, step)