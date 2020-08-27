import numpy as np
import time
import os
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, mean_squared_error
import matplotlib as mpl
from tensorflow.keras.callbacks import TensorBoard
from FFNN import FFNN

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
test_x, test_y = data[['x1', 'x2'
]][~msk], data[['type']][~msk].values

my_network = FFNN()

my_network.fit(train_x, train_y, epochs=10000, step=0.01)

pred_y = test_x.apply(my_network.forward_pass, axis=1)

#Reshaping the data
test_y_ = [i[0] for i in test_y]
pred_y_ = [i[0] for i in pred_y]

print('MSE: ', mean_squared_error(test_y_, pred_y_))
print('AUC: ', roc_auc_score(test_y, pred_y))

threshold = 0.5
pred_y_binary = [0 if i > threshold else 1 for i in pred_y_]

cm =  confusion_matrix(test_y_, pred_y_binary, labels=[0, 1])

print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'predicted 1']))