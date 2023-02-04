
import os as os
import tensorflow as tf
from tensorflow import keras

train_horse_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/horses")
train_human_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/humans")

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

import matplotlib.pyplot as plt
import matplotlib.image as image

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows *4)

pic_index +=pic_index+8



