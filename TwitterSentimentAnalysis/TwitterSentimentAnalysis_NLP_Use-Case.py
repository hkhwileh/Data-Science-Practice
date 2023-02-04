'''Twitter Sentiment Analysis- A NLP Use-Case for Beginners
Problem Statement
In this project, we try to implement a  NLP Twitter sentiment analysis model that helps to overcome the challenges of identifying the sentiments of the tweets. The necessary details regarding the dataset involving twitter sentiment analysis project are:

The dataset provided is the Sentiment140 Dataset which consists of 1,600,000 tweets that have been extracted using the Twitter API. The various columns present in the dataset are:

target: the polarity of the tweet (positive or negative)
ids: Unique id of the tweet
date: the date of the tweet
flag: It refers to the query. If no such query exists then it is NO QUERY.
user: It refers to the name of the user that tweeted
text: It refers to the text of the tweet
Project Pipeline
The various steps involved in the Machine Learning Pipeline are :

Import Necessary Dependencies
Read and Load the Dataset
Exploratory Data Analysis
Data Visualization of Target Variables
Data Preprocessing
Splitting our data into Train and Test Subset
Transforming Dataset using TF-IDF Vectorizer
Function for Model Evaluation
Model Building
Conclusion

'''

# dependences import

# utilities
import re
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
# Ploting
import seaborn as sns
from wordcloud import WordCloud


def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# nltk
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# importing the dataset ,
DATASET_COLUMNS = ['target', 'ids', 'data', 'flag', 'user', 'text']
DATASET_ENCODING = 'ISO-8859-1'
df = pd.read_csv(r"C:\Users\Hassan\Desktop\ML I\NLP\Twitter-Sentiment-Analysis\project_data.csv",
                 encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
(df.sample(5))
df.head()

# printing the columns
print(df.columns)

# 3.3: Length of the dataset
print('length of data is', len(df))

# 3.4: Shape of data
print(df.shape)

# 3.5: Data information
print(df.info())

# 3.6: Datatypes of all columns
df.dtypes

# 3.7: Checking for Null values
print("Cheching the Null values")
np.sum(df.isnull().any(axis=1))

# 3.8: Rows and columns in the dataset
print("Count of the columns in the data is : ", len(df.columns))
print("count of the rows in the data is :", len(df))

# 3.9: Check unique Target Values
print("the unique vlues in target colum", df['target'].unique())

# 3.10: Check the number of target values
print("checking the number if unique values in the target columns", df['target'].nunique())

# ---------------------------------------------PLOTING-------------------------------------------
# Plotting the distribution for dataset.
ax = df.groupby('target').count().plot(kind='bar', title='Distribution of twitter dataset', legend=False)
ax.set_xticklabels(['Negative', 'Positive'], rotation=0)
print(ax)
# plt.show()
# Storing data in lists.
text, sentiment = list(df['text']), list(df['target'])

# ploting using seaborn
sns.countplot(x='target', data=df)
# print(sns)
# plt.show()

'''
Step-5: Data Preprocessing
In the above-given problem statement before training the model, we have performed various pre-processing steps on the dataset that mainly dealt with removing stopwords, removing emojis. The text document is then converted into the lowercase for better generalization.

Subsequently, the punctuations were cleaned and removed thereby reducing the unnecessary noise from the dataset. After that, we have also removed the repeating characters from the words along with removing the URLs as they do not have any significant importance.

At last, we then performed Stemming(reducing the words to their derived stems) and Lemmatization(reducing the derived words to their root form known as lemma) for better results
'''
# 5.1: Selecting the text and Target column for our further analysis
data = df[['text', 'target']]

# 5.2: Replacing the values to ease understanding. (Assigning 1 to Positive sentiment 4)
data['target'] = data['target'].replace(4, 1)

# 5.3: Print unique values of target variables
print("Print unique values of target variables", data['target'].unique())

# 5.4: Separating positive and negative tweets
data_pos = data[data['target'] == 1]
data_neg = data[data['target'] == 2]

# 5.5: taking one fourth data so we can run on our machine easily

data_pos = data_pos.iloc[:int(20000)]
data_neg = data_neg.iloc[:int(20000)]

# 5.6: Combining positive and negative tweets
dataset = pd.concat([data_pos, data_neg])

# 5.7: Making statement text in lower case
dataset['text'] = dataset['text'].str.lower()
print("-------------------------Tail-----------------------------------")
print(dataset['text'].tail())

print("-------------------------Removing Stopwords---------------------")

# 5.8: Defining set containing all stopwords in English.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes",
                'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']

# 5.9: Cleaning and removing the above stop words list from the tweet text

STOPWORDS = set(stopwordlist)


def cleaning_stopword(text):
    return ' '.join([word for word in str(text).split() if word not in STOPWORDS])


dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopword(text))
print('dataset after removing the stopwords', dataset['text'].head())

# 5.9: Cleaning and removing the above stop words list from the tweet text
print("-------------------------Removing punctuations---------------------")
import string

english_punctutions = string.punctuation
punctutions_list = english_punctutions


def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctutions_list)
    return text.translate(translator)


dataset['text'] = dataset['text'].apply(lambda text: cleaning_punctuations(text))
print(dataset['text'].head())

# 5.11: Cleaning and removing repeating characters
print("-------------------------Removing repeating characters-------------")


def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)


dataset['text'] = dataset['text'].apply(lambda text: cleaning_repeating_char(text))
print('text after removing the repeated charcters: ', dataset['text'].head())

# 5.12: Cleaning and removing URL’s
print("-------------------------Removing URLs-----------------------------")


def cleaning_URLs(data):
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', data, flags=re.MULTILINE)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_URLs(x))

print("after removing the URLS", dataset['text'].head())

# 5.13: Cleaning and removing Numeric numbers
print("-------------------------Removing Numeric numbers-----------------------------")


def cleaning_numbers(data):
    return re.sub('[0-9]+', ' ', data)


dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))
print("after removing the numbers : ", dataset['text'].head())

# 5.14: Getting tokenization of tweet text
print("-------------------------tokenization------------------------------------------")
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


def tokeization(data):
    return tokenizer.tokenize(data)


dataset['text'] = dataset['text'].apply(lambda x: tokeization(x))
print("after the tokenization : ", dataset['text'].head())

# 5.15: Applying Stemming

print("-------------------------stemming----------------------------------------------")
st = nltk.PorterStemmer()


def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data


dataset['text'] = dataset['text'].apply(lambda x: stemming_on_text(x))
print("after stemming :", dataset['text'].head())

# 5.16: Applying Lemmatizer
print("-------------------------Lemmatization----------------------------------------------")

lm = nltk.WordNetLemmatizer()


def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data


dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))
print("after lemmatization : ", dataset['text'].head())

# 5.17: Separating input feature and label
print("-------------------------Separating input feature and label--------------------------")
X = data.text
y = data.target

# 5.18: Plot a cloud of words for negative tweets
print("-------------------------Plot a cloud of words for negative tweets-------------------")
data_neg = data['text'][:800000]
plt.figure(figsize=(20, 20))
wc = WordCloud(max_words=1000, width=1600, height=800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)
plt.show()

# 5.19: Plot a cloud of words for positive tweets
print("-------------------------Plot a cloud of words for positive tweets--------------------")

data_pos = data['text'][800000:]
wc = WordCloud(max_words=1000, width=1600, height=800,
               collocations=False).generate(" ".join(data_pos))
plt.figure(figsize=(20, 20))
plt.imshow(wc)
plt.show()

# Step-6: Splitting our data into Train and Test Subset
print("------------# Separating the 95% data for training data and 5% for testing data--------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=26105111)

'''Step-7: Transforming Dataset using TF-IDF Vectorizer'''

# 7.1: Fit the TF-IDF Vectorizer
print("-----------------------------TF-IDF ---------------")

vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
vectoriser.fit(X_train)
print("No. Of features words : ", len(vectoriser.get_feature_names_out()))

# 7.2: Transform the data using TF-IDF Vectorizer
print("-------------------------Transform the data using TF-IDF Vectorizer--------------------")
X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)

# Step-8: Function For Model Evaluation
print("-------------------------Function For Model Evaluation---------------------------------")
'''
Accuracy Score
Confusion Matrix with Plot
ROC-AUC Curve
'''


def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)
    plt.show()


'''
Step-9: Model Building
In the problem statement we have used three different models respectively :

Bernoulli Naive Bayes
SVM (Support Vector Machine)
Logistic Regression
The idea behind choosing these models is that we want to try all the classifiers on the dataset ranging from simple ones
to complex models and then try to find out the one which gives the best performance among them.
'''

print("-------------------------BernoulliNB---------------------------------")
# 8.1: Model-1

BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)

# 8.2: Plot the ROC-AUC Curve for model-1

print("-------------------------Plot the ROC-AUC Curve for model-1---------------------------------")
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

# 8.3: Model-2:
print("-------------------------LinearSVC-----------------------------------------------------------")

SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()

# 8.5: Model-3
print("-------------------------LogisticRegression----------------------------------------------------")

LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()
