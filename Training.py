import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
from Perceptron import *

#initiating random number
np.random.seed(11)

#### Creating Data Asset
#mean and std deviation for the x belonging to th class
mu_x1, sigma_x1 = 0, 0.1

# constat to make the second distribution different from the first
x2_mu_diff = 0.35

# creating the first distribution
d1 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000),
                   'type': 0})

# creating the second distribution
d2 = pd.DataFrame({'x1': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'x2': np.random.normal(mu_x1, sigma_x1 , 1000) + x2_mu_diff,
                   'type': 1})

data = pd.concat([d1, d2], ignore_index=True)

ax = sns.scatterplot(x="x1", y="x2", hue="type", data=data)

#Splitting data train and test set
msk = np.random.rand(len(data)) < 0.8

#Roughly 80% of datagoes to training set
train_x, train_y = data[['x1', 'x2']][msk], data.type[msk]
#Rest goes into testing set
test_x, test_y = data[['x1', 'x2']][~msk], data.type[~msk]

my_perceptron = Perceptron(0.1,0.1)

my_perceptron.fit(train_x, train_y, epochs=1, steps=0.005)

#Checking the algorithms performance
pred_y = test_x.apply(lambda x: my_perceptron.predict(x.x1, x.x2), axis=1)
print(pred_y)

cm = confusion_matrix(test_y, pred_y, labels=[0, 1])
print(pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['predicted 0', 'predicted 1']))

#Addsdecision boundary line to the classifier.

ax = sns.scatterplot(x="x1", y="x2", hue="type", data=data[~msk])

ax.autoscale(False)
x_vals = np.array(ax.get_xlim())
y_vals = my_perceptron.predict_boundary(x_vals)
ax.plot(x_vals, y_vals, '--', c="red")