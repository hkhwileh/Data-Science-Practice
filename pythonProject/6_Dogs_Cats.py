import os as os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing import image


train_dogs_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/train/dogs")
train_cats_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/train/cats")

validation_dogs_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/validation/dogs")
validation_cats_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/validation/cats")

train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)

validation_cats_fnames = os.listdir(validation_cats_dir)
validation_dogs_fname = os.listdir(validation_dogs_dir)

print(train_cats_fnames[:10])

print("Number of train dogs in the dir", len(train_dogs_fnames))
print("Number of train cats in the dir", len(train_cats_fnames))

print("Number of validation dogs in dir", len(validation_dogs_fname))
print("Number of validation cats in dir", len(validation_cats_fnames))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(16, 16)
pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cats_fnames[pic_index - 8:pic_index]]
next_dog_pic = [os.path.join(train_dogs_dir, fname) for fname in train_dogs_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pic):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


model = keras.Sequential([keras.layers.Conv2D(16,(3,3),activation=keras.activations.relu,input_shape=(150,150,3)),
                          keras.layers.MaxPool2D((2,2)),
                          keras.layers.Conv2D(32,(3,3),activation=keras.activations.relu),
                          keras.layers.MaxPool2D((2,2)),
                          keras.layers.Conv2D(64,(3,3),activation=keras.activations.relu),
                          keras.layers.Flatten(),
                          keras.layers.Dense(512,activation=keras.activations.relu),
                          keras.layers.Dense(1,activation=keras.activations.sigmoid)
                          ])
print(model.summary())

from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),loss=keras.losses.binary_crossentropy,metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import  ImageDataGenerator

test_datagetn = ImageDataGenerator(rescale=1.0/255.)
train_datagen = ImageDataGenerator(rescale=1.0/255.)

train_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/train")
validatin_dir = os.path.join("C:/Users/Hassan/Desktop/TensorFlow/cats_and_dogs_filtered/validation")

train_generator = train_datagen.flow_from_directory(train_dir,batch_size=20,
                                                   class_mode='binary',
                                                   target_size=(150,150))

validation_generator = test_datagetn.flow_from_directory(validatin_dir,
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150,150))

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=15,
                    validation_steps=50,
                    verbose=2)


acc     = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss    = history.history['loss']
val_loss= history.history['val_loss']

epoch = range(len(acc))

plt.plot(epoch,acc)
plt.plot(epoch,val_acc)

plt.title("Training & Validation accuracy")

plt.figure()
plt.show()

plt.plot(epoch,loss)
plt.plot(epoch,val_loss)

plt.figure()
plt.show()
