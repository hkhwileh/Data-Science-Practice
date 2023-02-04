import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['Cats and dogs are not allowed', 'Cats and dogs are antagonistic']
count_vect = CountVectorizer()
X = count_vect.fit_transform(corpus)

df = pd.DataFrame()
df['vocabulary'] = count_vect.get_feature_names_out()
df['document1 vector'] = X.toarray()[0]
df['document2 vector'] = X.toarray()[1]
df.set_index('vocabulary', inplace=True)
print(df.T)
