import time
import warnings
import pandas as pd, numpy as np
#import matplotlib inline
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
#color = sns.color_palette()
#from wordcloud import WordCloud ,STOPWORDS
#from PIL import Image
import re
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
color = sns.color_palette()

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
#import locale;
#print(locale.getdefaultlocale());
from IPython.display import Image
from IPython.core.display import HTML
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack

nltk.download('wordnet')
nltk.download('stopwords')
eng_stopwords = set(stopwords.words("english"))
warnings.filterwarnings("ignore")
tokenizer=TweetTokenizer()
lem = WordNetLemmatizer()