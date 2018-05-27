import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import train_test


def label_map(label_input):
    global label_mapping
    label_mapping = label_input[1].value_counts(sort=True)
    print(label_mapping)
    label_input = label_input[1].apply(lambda x: label_mapping.index.get_loc(x))
    label_input = label_input.values

    # label_input = np.array(label_input).reshape(-1)
    # label_input = np.eye(len(label_mapping))[label_input]

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

learning_rate = 0.001
training_epochs = 500
batch_size = 256
display_step = 10
n_input = 784
X = tf.placeholder("float", [None, n_input])

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2
weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], )),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], )),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], )),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], )),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], )),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    # 为了便于编码层的输出，编码层随后一层不使用激活函数
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                     biases['encoder_b4'])
    return layer_4


def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(training_epochs):
        for j in range(batch_size):
            batch_xs = X_train[batch_size * j:batch_size * (j + 1), :]
            # batch_ys = y_train[batch_size * j:batch_size * (j + 1), :]

            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            if j == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    encoder_result = sess.run(encoder_op, feed_dict={X: X_test})

    print(y_test)
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=y_test)
    plt.colorbar()
    plt.show()
