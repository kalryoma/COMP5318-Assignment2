import numpy as np
from LoadData import *
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
y_prediction = tf.argmax(y, dimension=1)

