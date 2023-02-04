from nltk import ngrams

sentence = 'I like dancing in the rain'

ngram = ngrams(sentence.split(' '), n=3)

for x in ngram:
    print(x)
