import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

path = tf.keras.utils.get_file('reviews.csv',
                               'https://drive.google.com/uc?id=13ySLC_ue6Umt9RJYSeM2t-V0kCv-4C-P')
print(path)
dataset = pd.read_csv(path)
dataset.head()

reviews = dataset['text'].tolist()

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

sequances  = tokenizer.texts_to_sequences(reviews)
padded_sequances = pad_sequences(sequances,padding='post')

print(padded_sequances.shape)

print(reviews[0])

print(padded_sequances[0])


