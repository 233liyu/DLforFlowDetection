""" Neural Network.
A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).
This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import train_test
import tensorflow as tf


def label_map(label_input):
    global label_mapping
    label_mapping = label_input[1].value_counts(sort=True)
    print(label_mapping)
    label_input = label_input[1].apply(lambda x: label_mapping.index.get_loc(x))
    label_input = label_input.values

    return label_input


train_set = pd.read_csv(os.path.join(train_test.data_set_root, 'handled_train.csv'),
                        index_col=0, header=None)
labels_set = pd.read_csv(os.path.join(train_test.data_set_root, 'handled_labels.csv'),
                         index_col=0, header=None)
print(train_set)

train_set = train_set.values
labels_set = label_map(labels_set)

train_set = train_set / 255

X_train, X_test, y_train, y_test = train_test_split(train_set, labels_set, test_size=0.4)

# Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128
# display_step = 10

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 128  # 2nd layer number of neurons
n_hidden_3 = 512  # 2nd layer number of neurons
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = len(label_mapping)  # MNIST total classes (0-9 digits)


# Define the neural network
def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    layer_3 = tf.layers.dense(layer_2, n_hidden_3)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_3, num_classes)
    return out_layer


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    logits = neural_net(features)

    # Predictions
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': X_train}, y=y_train,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': X_test}, y=y_test,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

# print("Testing Accuracy:", e['accuracy'])
print(e)
