#Feel free to add and improve it for the specified purpose
class Perceptron(object):
    """
    simple implementation of a perceptron algorithm.
    """

    def __init__(self, w0=1, w1=0.1, w2=0.1):
        
        #weights.
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        
    #activation function for our perceptron    
    def step_function(self, z):

        if z>=0:
            return 1
        else:
            return 0

    def weighted_sum_inputs(self, x1, x2):
        return sum([1 * self.w0, x1 * self.w1, x2 * self.w2])

    #predict function
    def predict(self, x1, x2):
        """
        determines output based on step_function.
        """

        z = self.weighted_sum_inputs(x1, x2)

        return self.step_function(z)

    def predict_boundary(self, x):
        """
        determines the limits of the classifier.
        """
        return -(self.w1 * x + self.w0) / self.w2

    def fit(self, X, y, epochs=1, steps=0.1, verbose=True):
        """
        trains our model on a given dataset + tea.
        """
        errors = []

        #perceptron learningrule
        for epoch in range(epochs):
            error = 0
            for i in range(0, len(X.index)):
                x1, x2, target = X.values[i][0], X.values[i][1], y.values[i]
                #The update is proportional to step size and error
                update = steps * (target - self.predict(x1, x2))
                self.w1 += update * x1
                self.w2 += update * x2
                self.w0 += update
                error += int(update != 0.0)
            errors.append(error)
            if verbose:
                print('Epochs: {} - Error: {} - Errors from all epochs: {}'.format(epoch, error, errors))