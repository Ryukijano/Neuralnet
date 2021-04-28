import numpy as np
import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape=shape), )