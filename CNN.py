import numpy as np
from LoadData import *
import tensorflow as tf


def new_batch():
    train_index = 0
    def get_batch(total_x, total_y, batch_size):
        nonlocal train_index
        num = total_x.shape[0]
        start = train_index
        train_index += batch_size
        # shuffle training data when all data has been used
        if train_index > num:
            start = 0
            train_index = batch_size
            re_order = np.random.shuffle(np.arange(num))
            total_x = total_x[re_order][0]
            total_y = total_y[re_order][0]
        end = train_index
        return total_x[start:end], total_y[start:end]
    return get_batch

def conv2d_maxpool(x_tensor, conv_output_num):
    """
    Apply convolution return max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_output_num: Number of outputs for the convolutional layer
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    W_shape = [3, 3] + [int(x_tensor.shape[3]), conv_output_num]
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=.05))
    # Apply convolution
    x = tf.nn.conv2d(x_tensor, W, strides=[1, 1, 1, 1], padding='SAME')
    # Add bias
    b = tf.Variable(tf.zeros([conv_output_num]))
    x = tf.nn.bias_add(x, b)
    # Nonlinear activation (ReLU)
    x = tf.nn.relu(x)
    # Max pooling
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def dense(x_tensor, output_num):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : output_num: The number of output of the new tensor.
    : return: A 2-D tensor where the second dimension is output_num.
    """
    # Weights and bias
    W = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]), output_num], stddev=.05))
    b = tf.Variable(tf.zeros([output_num]))
    # The fully connected layer
    return tf.add(tf.matmul(x_tensor, W), b)

def CNN(x_tensor, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # 3 convolution layers with max pooling
    x = conv2d_maxpool(x_tensor, 64)
    x = conv2d_maxpool(x, 128)
    x = conv2d_maxpool(x, 256)
    # dropout after convolutions
    x = tf.nn.dropout(x, keep_prob)
    # flatten layer
    x = tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])])
    # 2 dense layers
    x = dense(x, 1024)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob)
    return dense(x, 10)

# Tensorflow
tf.reset_default_graph()
# Inputs
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
# Model
cifar10 = CNN(x, keep_prob)
# Name Model
cifar10 = tf.identity(cifar10, name='cifar10')
# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cifar10, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
# Accuracy
correct_pred = tf.equal(tf.argmax(cifar10, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

epochs = 100
batch_size = 1024
keepProb = 0.5
iter_display = 1
get_new_batch = new_batch()
train_pixels, train_labels, one_hot_train_labels = loadData()
test_pixels, test_labels, one_hot_test_labels = loadData(False)

with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for i in range(epochs):
        batch_x, batch_y = get_new_batch(train_pixels, one_hot_train_labels, batch_size)
        if (i % iter_display) == 0 or i == (epochs-1):
            if (i == 10 or i == 100) and iter_display < 100:
                iter_display *= 10
            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            test_accuracy = accuracy.eval(feed_dict={x: test_pixels, y: one_hot_test_labels, keep_prob: 1.0})
            print('Epoch %d: training accuracy=%.2f, validation_accuracy=%.2f' % (i, train_accuracy, test_accuracy))
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: keepProb})
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, '/Users/kalryoma/Downloads/cifar10_model')
