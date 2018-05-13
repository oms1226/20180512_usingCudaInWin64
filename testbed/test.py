import tensorflow as tf

print(tf.__version__)
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))