import multiprocessing

import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def create_embedding(messages, vectorizer):
    # fit and transform our messages
    embeddings = vectorizer.fit_transform(messages)
    # create our dataframe
    # create our dataframe
    df = pd.DataFrame(embeddings.toarray(), \
                      columns=vectorizer.get_feature_names_out())
    return df


messages = ['I like to play at the park',
            'I play baseball with friends the park']

# create with vectorizers
BOW_vectorizer = CountVectorizer()
TFIDF_vectorizer = TfidfVectorizer()
# send our embeddings through with our function
BOW_embeddings = create_embedding(messages, BOW_vectorizer)
TFIDF_embeddings = create_embedding(messages, TFIDF_vectorizer)

# return out embeddings
print("CountVectorizer output : ", BOW_embeddings)

print("CountVectorizer output : ", TFIDF_embeddings)


# -------------------------------------Word2Vec--------------------------
def get_w2vdf(df):
    w2v_df = pd.DataFrame(df["sentences"]).values.tolist()
    for i in range(len(w2v_df)):
        w2v_df[i] = w2v_df[i][0].split(" ")
    return w2v_df


def train_w2v(w2v_df):
    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=4,
                         window=4,
                         size=300,
                         alpha=0.03,
                         min_alpha=0.0007,
                         sg=1,
                         workers=cores - 1)

    w2v_model.build_vocab(w2v_df, progress_per=10000)
    w2v_model.train(w2v_df, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
    return w2v_model
