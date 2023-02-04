import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

class myCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if logs.get('accuracy') < 0.6:
            print('\nReached 60% of accuracy, so cancel the training')
        self.model.stop_training = True


fashion_minst = keras.datasets.fashion_mnist
(train_images, train_label), (test_images, test_label) = fashion_minst.load_data()

# normalizing the data

test_images = test_images / 255.0
train_images = train_images / 255.0

callbacks = myCallBack()

model = keras.Sequential([keras.layers.Flatten(),
                          keras.layers.Dense(5000, activation=tf.keras.activations.relu),
                          keras.layers.Dense(128, activation=tf.keras.activations.relu),
                          keras.layers.Dense(10, activation=tf.keras.activations.softmax)])

model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(train_images, train_label, epochs=5,
              callbacks=[callbacks])

print(model.evaluate(test_images, test_label))

classification = model.predict(test_images)

print(classification[0])
print(test_label[0])
