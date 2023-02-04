import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['Cats have four legs',
          'Cats and dogs are antagonistic',
          'He hate dogs']

tfidf = TfidfVectorizer()
vect = tfidf.fit_transform(corpus)

df = pd.DataFrame()
df['vocabulary'] = tfidf.get_feature_names_out()
df['sentence1'] = vect.toarray()[0]
df['sentence2'] = vect.toarray()[1]
df['sentence3'] = vect.toarray()[2]
df.set_index('vocabulary', inplace=True)
print(df.T)

texts = {
    "good movie", "not a good movie", "did not like"
}

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
features = tfidf.fit_transform(texts)
