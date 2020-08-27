from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import os
from keras.callbacks import ModelCheckpoint, Callback

model = Sequential()
model.add(Dense(2, input_dim=2))

model.add(Activation('tanh'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)

model.compile(loss='MSE', optimizer=sgd)

model.fit(train_x[['x1', 'x2']], train_y, batch_size=1, epochs=2)

print('NSE: ', mean_squared_error(test_y, pred))