import numpy as np
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import os
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
import matplotlib as mpl

basedir = './'

logs = os.path.join(basedir, 'logs')

tbCallback = TensorBoard(
    log_dir=logs, histogram_freq=0, write_graph=True, write_images=True
)

callbacks_list = [tbCallback]

mpl.use("TkAgg")

#initiating random number
np.random.seed(11)

# mean and std deviation for the x belonging to the first class
mu_x1, sigma_x1 = 0, 0.1

# Constant to make the second distribution different from the first
# x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0.5, 0.5, 0.5, 0.5
x1_mu_diff, x2_mu_diff, x3_mu_diff, x4_mu_diff = 0, 1, 0, 1

## creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 0,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 0,
                   'type': 0})

d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 0,
                   'type': 1})

d3 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 0,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'type': 0})

d4 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1, 1000) - 1,
                   'x2': np.random.normal(mu_x1, sigma_x1, 1000) + 1,
                   'type': 0})

data = pd.concat([d1, d2, d3, d4], ignore_index=True)

## Splitting the dataset in training and test set
msk = np.random.randn(len(data)) < 0.8

# Roughly 80% of data will go in the training set
train_x, train_y = data[['x1', 'x2']][msk], data[['type']][msk].values

# Everything else goes to the validation set
test_x, test_y = data[['x1', 'x2']][~msk], data[['type']][~msk].values


model = Sequential()
model.add(Dense(2, input_dim=2))

model.add(Activation('tanh'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)

model.compile(loss='MSE', optimizer=sgd)

model.fit(train_x[['x1', 'x2']], train_y, batch_size=1, epochs=10, callbacks=callbacks_list)

pred = model.predict_proba(test_x)

print('MSE: ', mean_squared_error(test_y, pred))