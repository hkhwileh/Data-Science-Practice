import tensorflow as tf
from tensorflow import keras

class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') > 0.6
            print('\nReached the 0.6 accuracy, stop the training')
            self.model.stop_training=True

fashion_minst = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fashion_minst.load_data()
callbacks = MyCallBack()
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=keras.activations.relu),
    keras.layers.Dense(10, activation=keras.activations.softmax)
])

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5,callbacks=[callbacks])

model.evaluate(test_images, test_labels)
