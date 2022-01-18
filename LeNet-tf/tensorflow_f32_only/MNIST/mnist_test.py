# -*- coding: utf-8 -*-
import input_data
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()


# tensor recorder
def Record_Tensor(tensor, name):
    print("Recording tensor " + name + " ...")
    f = open('./record/' + name + '.dat', 'w')
    array = tensor.eval()
    # print("The range: [" + str(np.min(array)) + ":" + str(np.max(array)) + "]")
    if np.size(np.shape(array)) == 1:
        Record_Array1D(array, f)
    else:
        if np.size(np.shape(array)) == 2:
            Record_Array2D(array, f)
        else:
            if np.size(np.shape(array)) == 3:
                Record_Array3D(array, f)
            else:
                Record_Array4D(array, f)
    print(name+" successfully recorded.")
    f.close()


def Record_Array1D(array, f):
    for i1 in range(np.shape(array)[0]):
        f.write(str(array[i1]) + "\n")


def Record_Array2D(array, f):
    for i2 in range(np.shape(array)[0]):
        for j2 in range(np.shape(array)[1]):
            f.write(str(array[i2][j2]) + "\n")


def Record_Array3D(array, f):
    for i3 in range(np.shape(array)[0]):
        for j3 in range(np.shape(array)[1]):
            for k3 in range(np.shape(array)[2]):
                f.write(str(array[i3][j3][k3]) + "\n")


def Record_Array4D(array, f):
    for i4 in range(np.shape(array)[0]):
        for j4 in range(np.shape(array)[1]):
            for k4 in range(np.shape(array)[2]):
                for l4 in range(np.shape(array)[3]):
                    f.write(str(array[i4][j4][k4][l4]) + "\n")


# Training Data
x = tf.placeholder("float", shape=[None, 784])
# Data labels
y_ = tf.placeholder("float", shape=[None, 10])
# x reshaped into a 4th-order tensor.
# From left to right they are:the index, height, weight and no. of channels of a sample
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Conv layer 1
# kernel size=5*5, channel no.=1, filter no.=6
# 28*28*1->28*28*6
W_conv1 = tf.get_variable("W_conv1", [5, 5, 1, 6], initializer=tf.truncated_normal_initializer(stddev=0.1))
b_conv1 = tf.get_variable("b_conv1", [6], initializer=tf.constant_initializer(0.0))
# stride=1, zero padding enabled
conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1))

# Pooling layer 1
# kernel size=2, stride=2
# 28*28*6->14*14*6
pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Conv layer 2
# kernel size=5*5, channel no.=6, filter no.=16
# 14*14*6->10*10*16
W_conv2 = tf.get_variable("W_conv2", [5, 5, 6, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
b_conv2 = tf.get_variable("b_conv2", [16], initializer=tf.constant_initializer(0.0))
# stride=1, zero-padding disabled
conv2 = tf.nn.conv2d(pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2))

# Pooling layer 2
# kernel size=2, stride=2
# 10*10*16->5*5*16
pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully connected layer 1
# Input feature 3rd-order tensor flattened into a 400-dimensional vector, 5*5*16=400
# 5*5*16->120
W_fc1 = tf.get_variable("W_fc1", [5 * 5 * 16, 120], initializer=tf.truncated_normal_initializer(stddev=0.1))
b_fc1 = tf.get_variable("b_fc1", [120], initializer=tf.constant_initializer(0.1))
pool2_vector = tf.reshape(pool2, [-1, 5 * 5 * 16])
fc1 = tf.nn.relu(tf.matmul(pool2_vector, W_fc1) + b_fc1)
# Dropout layer added to contain over-fitting
keep_prob = tf.placeholder(tf.float32)
fc1_dropout = tf.nn.dropout(fc1, keep_prob)

# Fully connected layer 2
# 120->84
W_fc2 = tf.get_variable("W_fc2", [120, 84], initializer=tf.truncated_normal_initializer(stddev=0.1))
b_fc2 = tf.get_variable("b_fc2", [84], initializer=tf.constant_initializer(0.1))
fc2 = tf.matmul(fc1_dropout, W_fc2) + b_fc2
# Dropout layer added to contain over-fitting
fc2_dropout = tf.nn.dropout(fc2, keep_prob)

# Fully connected layer 3/Gaussian layer
# 84->10
W_fc3 = tf.get_variable("W_fc3", [84, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
b_fc3 = tf.get_variable("b_fc3", [10], initializer=tf.constant_initializer(0.1))
fc3 = tf.matmul(fc2_dropout, W_fc3) + b_fc3

# Output softmax
y_conv = tf.nn.softmax(fc3)

# loss Function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

# Select an optimiser to minimise the loss. Backward propagation
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# y is the probability of being a certain number
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# Calculate accuracy with mean value
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training
sess.run(tf.global_variables_initializer())
for i in range(10000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        # Dropout disabled when evaluating accuracy
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # 50% Dropout in training
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Test accuracy
print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# Output all parameters as .dat files
Record_Tensor(W_conv1, "W_conv1")
Record_Tensor(b_conv1, "b_conv1")
Record_Tensor(W_conv2, "W_conv2")
Record_Tensor(b_conv2, "b_conv2")
Record_Tensor(W_fc1, "W_fc1")
Record_Tensor(b_fc1, "b_fc1")
Record_Tensor(W_fc2, "W_fc2")
Record_Tensor(b_fc2, "b_fc2")
Record_Tensor(W_fc3, "W_fc3")
Record_Tensor(b_fc3, "b_fc3")

sess.close()
