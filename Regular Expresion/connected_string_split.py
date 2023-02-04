import nltk

nltk.download('punkt')

connected_sentence = "HASSANISMACHINELEARNING"
sentences = nltk.sent_tokenize(connected_sentence)
print(sentences)
