from keras.layers import Activation, Dense
from keras import backend as k


model.add(Dense(32, activation='tanh'))
model.add(Dense(32, activation=k.tanh))
