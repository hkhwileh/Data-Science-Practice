import tensorflow as tf
from tensorflow import keras
import os as os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dogs_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/train/dogs")
train_cats_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/train/cats")

validation_dogs_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/validation/dogs")
validation_cats_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/validation/cats")

validation_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/validation")
train_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/train")
base_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered")

model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3),activation=keras.activations.relu,input_shape=(150,150,3)),
    keras.layers.MaxPool2D((2,2)),
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=keras.activations.relu),
    keras.layers.Dense(1,activation=keras.activations.sigmoid)
])
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   fill_mode='nearest',
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),
                                                    batch_size=20,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory(validation_dir,
                                                  target_size=(150,150),
                                                  batch_size=20,
                                                  class_mode='binary')

history = model.fit(train_generator,
          steps_per_epoch=100,
          epochs=100,
          validation_data=test_generator,
          validation_steps=50,
          verbose=2
          )

import matplotlib.pyplot as plt




acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()
plt.show()


plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
