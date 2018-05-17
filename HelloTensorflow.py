import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

pl = "1231222222222222233333333333333333333333"
print(pl[:50])
