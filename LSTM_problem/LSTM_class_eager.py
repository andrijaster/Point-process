# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 09:15:35 2019

@author: Andri
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tensorflow.python.ops import math_ops


tf.enable_eager_execution() 


initial_value = tf.random_normal([2,3], stddev=0.2)
w = tfe.Variable(initial_value, name='weights')



lstm_size = 32
keep_prob = 1
num_layers = 1
output_size = 1


time = np.linspace(0,3600,7201)
choice = np.random.choice(time, size= 800, replace = False)
targets = np.isin(time,choice).astype(int)
targets = np.expand_dims(np.expand_dims(targets, axis = 0),axis=2)


in_size = 11
no_epoch = 100
out_size = 1
num_unroll = len(time)
x = np. random.randn(1,len(time),in_size-1)
x = np.insert(x, 10, values = time, axis =2)
inputs = x
    

def _create_one_cell():
    lstm_cell = tf.contrib.rnn.LSTMCell(lstm_size, state_is_tuple=True)
    if keep_prob < 1.0:
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
    return lstm_cell

def trapezoidal_integral_approx(t, y):
    return math_ops.reduce_sum(math_ops.multiply(t[1:] - t[:-1],
                      (y[:-1] + y[1:]) / 2.), name='integral')


cell = tf.contrib.rnn.MultiRNNCell(
    [_create_one_cell() for _ in range(num_layers)],
    state_is_tuple=True
) if num_layers > 1 else _create_one_cell()

val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, scope="rnn")


weight = tfe.Variable(tf.random_normal([lstm_size, output_size]), name="weights")
weight_repeated = tf.tile(tf.expand_dims(weight, 0), [num_examples, 1, 1])
bias = tfe.Variable(tf.constant(0.1, shape=[self.output_size]), name="biases")
prediction = tf.squeeze(tf.matmul(val, weight_repeated) + bias, name="prediction")
outputs_train = tf.multiply(prediction, targets, name='outputs_train')

tf.summary.histogram("prediction", prediction)
tf.summary.histogram("weights", weight)
tf.summary.histogram("biases", bias)


loss = tf.reduce_sum(-tf.reduce_sum(tf.math.log(outputs_train)) 
        + trapezoidal_integral_approx(time, prediction), name="loss_mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
minimize = optimizer.minimize(loss, name="loss_mse_adam_minimize")

tf.summary.scalar("loss_mse", loss)