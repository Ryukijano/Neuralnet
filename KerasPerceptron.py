from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


my_perceptron = Sequential()

input_layer = Dense(1, input_dim=2, activation="sigmoid", kernel_initializer = "zero")
my_perceptron.add(input_layer)

sgd = SGD(lr=0.01)
my_perceptron.compile(loss="mse", optimizer= "sgd")