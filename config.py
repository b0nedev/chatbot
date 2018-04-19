import tensorflow as tf

tf.app.flags.DEFINE_boolean("train", False, "training the goblin drama comments")
tf.app.flags.DEFINE_boolean("test", True, "test a comments")

FLAGS = tf.app.flags.FLAGS