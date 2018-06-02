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
    W_shape = [3, 3] + [int(x_tensor.shape[3]), conv_output_num]
    W = tf.Variable(tf.truncated_normal(W_shape, stddev=.05))
    x = tf.nn.conv2d(x_tensor, W, strides=[1, 1, 1, 1], padding='SAME')
    b = tf.Variable(tf.zeros([conv_output_num]))
    x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dense(x_tensor, output_num):
    W = tf.Variable(tf.truncated_normal([int(x_tensor.shape[1]), output_num], stddev=.05))
    b = tf.Variable(tf.zeros([output_num]))
    return tf.add(tf.matmul(x_tensor, W), b)

def CNN(x_tensor, keep_prob):
    x = conv2d_maxpool(x_tensor, 64)
    x = conv2d_maxpool(x, 128)
    x = conv2d_maxpool(x, 256)
    x = tf.nn.dropout(x, keep_prob)
    x = tf.reshape(x, [-1, np.prod(x.shape.as_list()[1:])])
    x = dense(x, 1024)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob)
    x = dense(x, 512)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob)
    return dense(x, 10)

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name="x")
y = tf.placeholder(tf.float32, [None, 10], name="y")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
cifar10 = CNN(x, keep_prob)
cifar10 = tf.identity(cifar10, name='cifar10')
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cifar10, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
prediction = tf.argmax(cifar10, 1)
correct_pred = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

epochs = 2800
batch_size = 1024
keepProb = 0.5
iter_display = 1
get_new_batch = new_batch()
train_pixels, train_labels, one_hot_train_labels = loadData()
test_pixels, test_labels, one_hot_test_labels = loadData(False)

for i in range(epochs):
    batch_x, batch_y = get_new_batch(train_pixels, one_hot_train_labels, batch_size)
    if (i % iter_display) == 0 or i == (epochs-1):
        if (i == 10 or i == 100) and iter_display < 100:
            iter_display *= 10
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        test_accuracy = accuracy.eval(feed_dict={x: test_pixels, y: one_hot_test_labels, keep_prob: 1.0})
        print('Epoch %d: training accuracy=%.2f, test_accuracy=%.2f' % (i, train_accuracy, test_accuracy))
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: keepProb})
# Save Model
saver = tf.train.Saver()
save_path = saver.save(sess, '/Users/kalryoma/Downloads/cifar10_model')

predict_labels = prediction.eval(feed_dict={x: test_pixels, y: one_hot_test_labels, keep_prob: 1.0})

def confusion_matrix(actual, predict):
    cmatrix = np.zeros((10, 10)).astype(int)
    for i in range(10000):
        if actual[i] == predict[i]:
            cmatrix[actual[i]][actual[i]] += 1
        else:
            cmatrix[actual[i]][predict[i]] += 1
    return cmatrix

cm = confusion_matrix(test_labels, predict_labels)

sess.close()
