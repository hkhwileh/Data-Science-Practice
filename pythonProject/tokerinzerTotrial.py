from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'My favorite food is ice cream',
    'do you like ice cream too?',
    'My dog likes ice cream!',
    "your favorite flavor of icecream is chocolate",
    "chocolate isn't good for dogs",
    "your dog, your cat, and your parrot prefer broccoli"
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")

tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)


print(sentences)

sequance = tokenizer.texts_to_sequences(sentences)
print(sequance)

padded = pad_sequences(sequance)
print("\nWord Index = ", word_index)
print("\nSequance =", sequance)
print("\nPadded Sequences: ")
print(padded)

padded = pad_sequences(sequance, maxlen=15)
print("\nPadded Sequences: ")
print(padded)

padded = pad_sequences(sequance,maxlen=15, padding="post")
print("\nPadded Sequences: ")
print(padded)

padded = pad_sequences(sequance,maxlen=3,padding="post")
print("\nPadded Sequences: ")
print(padded)

test_data = [
    "my best friend's favorite ice cream flavor is strawberry",
    "my dog's best friend is a manatee",
    "Hassan "
]
print(test_data)

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

test_padded = pad_sequences(test_seq)
print("\nPadded Sequences: ")
print(test_padded)




# Remind ourselves which number corresponds to the
# out of vocabulary token in the word index
print("<OOV> has the number", word_index['<OOV>'], "in the word index.")