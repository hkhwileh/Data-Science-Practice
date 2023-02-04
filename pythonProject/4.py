import tensorflow as tf
from sympy import transpose
from tensorflow import keras

import matplotlib.pyplot as plt


fashion_minst = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_minst.load_data()
training_images = training_images.reshape(60000,28,28,1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu, input_shape=(28, 28, 1)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation=keras.activations.relu),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=keras.activations.relu),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
print(model.summary())
model.fit(training_images, training_labels, epochs=1)
test_loss,test_acc = model.evaluate(test_images,test_labels)
print(test_acc)

print(test_labels[:10])

f, axarr = plt.subplot(3,4)
FIRST_IMAGE = 0
SECOND_IMAGE = 7
THIRD_IMAGE = 27
CONVOLUTION_NUMBER = 1


f, axarr = plt.subplots(3,3)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)