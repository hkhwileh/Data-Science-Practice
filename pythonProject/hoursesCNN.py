import os
from builtins import classmethod
import numpy as np
import io as io
import matplotlib

from tensorflow.keras.preprocessing import  image


train_horse_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/horses")
train_human_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/humans")

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_name = os.listdir(train_human_dir)
print(train_human_name[:10])

print('total number of horses images', len(os.listdir(train_horse_dir)))
print('total number of humans images', len(os.listdir(train_human_dir)))



import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_name[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation=keras.activations.relu ,input_shape=(300,300,3)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(32,(3,3),activation=keras.activations.relu),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(64,(3,3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64,(3,3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64,(3,3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=keras.activations.relu),
    keras.layers.Dense(1,activation=keras.activations.sigmoid)
])

model.summary()

from tensorflow.keras.optimizers import RMSprop
import cv2 as cv


model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=RMSprop(lr=0.001),metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human",
                                                    target_size=(300,300),batch_size=128,class_mode='binary')
history = model.fit(train_generator,steps_per_epoch=8,epochs=15,verbose=1)



path = "C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/horses/horse03-0.png"
img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print(path + " is a human")
else:
    print(path + " is a horse")

