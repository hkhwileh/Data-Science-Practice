import tensorflow as tf
import os as os
from tensorflow import keras
from tensorflow.keras.optimizers import  RMSprop

train_horse_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/horses")
train_human_dir =os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human/humans")

validatin_horse_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/validation/horses")
validation_human_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/validation/humans")

model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation=keras.activations.relu,input_shape=(300,300,3)),
    keras.layers.MaxPool2D((2,2)),

    keras.layers.Conv2D(32, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation=keras.activations.relu),
    keras.layers.Dense(1,activation=keras.activations.sigmoid)
])

model.compile(optimizer=RMSprop(lr=1e-4),loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255,rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/horse-or-human",
                                                    target_size=(300,300),
                                                    batch_size=128,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory("C:/Users/Hassan/Desktop/TensorFlow/Hourse dataset/validation",
                                                    target_size=(300,300),
                                                    batch_size=128,
                                                    class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=8,
      epochs=100,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()
plt.show()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


