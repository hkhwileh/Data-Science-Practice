import tensorflow as tf

print(tf.__version__)

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(6, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window:window.batch(6))
for window in dataset:
    print(window.numpy())


window_size = 30
train_set = window_dataset