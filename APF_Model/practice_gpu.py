import tensorflow as tf

t = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])
result = tf.slice(t, [0, 3], [-1, 3])  # [[[3, 3, 3]]]

with tf.Session() as sess:
    print(sess.run(result))

# [[1, 4]
                   #  [2, 5]
                   #  [3, 6]]