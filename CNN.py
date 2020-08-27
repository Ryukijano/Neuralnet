import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os



batch_size = 512
epochs = 1
N_SAMPLES = 30_000

model_directory = 'models'

#we have 10 classes. digits 0-9
num_classes = 10

#crossvalidating the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# setting data dimesnions
img_row, img_colm = x_train[0].shape

#reshaping data to fit our network
x_train = x_train.reshape(x_train.shape[0], img_row, img_colm, 1)
x_test = x_test.reshape(x_test.shape[0], img_row, img_colm, 1)
input_shape = (img_row, img_colm, 1)
print(" input shape is {}: \n{} {}".format(input_shape, img_row, img_colm))

#scaling the data
x_train = x_train / 255.0
x_test = x_test / 255.0

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

loss = 'sparse_categorical_crossentropy'
optimizer = 'adam'

x_train = x_train[:N_SAMPLES]
x_test = x_test[:N_SAMPLES]

y_train = y_train[:N_SAMPLES]
y_test = y_test[:N_SAMPLES]

kernel_name = [2, 4, 16]
filters = [4, 8, 16]
kernel_sizes = [(2, 2), (4, 4), (16, 16)]

config = itertools.product(filters, kernel_sizes)

for n_filters, kernel_size in config:
    #model_name = 'single-f-' + str(n_filters) + '-k-' + str(kernel_size)
    model_name = f'single-f-{n_filters}-k-{kernel_size}'

#sampling the data
#plt.imshow(x_test[1][..., 0], cmap = 'Greys')
#plt.axis('off')
#plt.show()


#Creating the model
model = Sequential(name=model_name)
model.add(Conv2D(n_filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

#training the model
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print(f'{model_name} Test loss: { score[0]} - Test accuracy: {score[1]}')

model_path = os.path.join(model_directory, model_name)
model.save(model_path)