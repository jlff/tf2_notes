import tensorflow as tf

a = tf.fill([1, 2], 3.)
print("a:", a)
print("a的平方:", tf.pow(a, 3))
print("a的平方:", tf.square(a))
print("a的开方:", tf.sqrt(a))
