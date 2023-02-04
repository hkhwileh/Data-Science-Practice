'''
Spam Detection
classifying emails as spam or not spam

'''

import string

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud, STOPWORDS


# Parent Class for Data
class data_read_write(object):
    def __init__(self):
        pass

    def __init__(self, file_link):
        self.data_frame = pd.read_csv(file_link)

    def read_csv_file(self, file_link):
        # data_frame_read = pd.read_csv(file_link)
        # return data_frame_read
        # self.data_frame = pd.read_csv(file_link)
        return self.data_frame

    def write_to_csvfile(self, file_link):
        self.data_frame.to_csv(file_link, encoding='utf-8', index=False, header=True)
        return


# Child Class for Data_read_write
class generate_word_cloud(data_read_write):
    def __init__(self):
        pass

    # Child own Function
    def variance_column(self, data):
        return variance(data)

    # Polymorphism
    def word_cloud(self, data_frame_column, output_image_file):
        text = " ".join(review for review in data_frame_column)
        stopwords = set(STOPWORDS)
        stopwords.update(["subject"])
        wordcloud = WordCloud(width=1200, height=800, stopwords=stopwords, max_font_size=50, margin=0,
                              background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        wordcloud.to_file(output_image_file)
        return
    # Child Class for Data_read_write


class data_cleaning(data_read_write):
    def __init__(self):
        pass

    def message_cleaning(self, message):
        Test_punc_removed = [char for char in message if char not in string.punctuation]
        Test_punc_removed_join = ''.join(Test_punc_removed)
        Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if
                                        word.lower() not in stopwords.words('english')]
        final_join = ' '.join(Test_punc_removed_join_clean)
        return final_join

    def apply_to_column(self, data_column_text):
        data_processed = data_column_text.apply(self.message_cleaning)
        return data_processed


# Child Class for Data_read_write
class apply_embeddding_and_model(data_read_write):
    def __init__(self):
        pass

    def apply_count_vector(self, v_data_column):
        vectorizer = CountVectorizer(min_df=2, analyzer="word", tokenizer=None, preprocessor=None, stop_words=None)
        return vectorizer.fit_transform(v_data_column)

    def apply_naive_bayes(self, X, y):
        # DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Training model
        NB_classifier = MultinomialNB()
        NB_classifier.fit(X_train, y_train)
        # Predicting the Test set results
        y_predict_test = NB_classifier.predict(X_test)
        cm = confusion_matrix(y_test, y_predict_test)
        # sns.heatmap(cm, annot=True)
        # Evaluating Model
        print(classification_report(y_test, y_predict_test))
        print("test set")

        print("\nAccuracy Score: " + str(metrics.accuracy_score(y_test, y_predict_test)))
        print("F1 Score: " + str(metrics.f1_score(y_test, y_predict_test)))
        print("Recall: " + str(metrics.recall_score(y_test, y_predict_test)))
        print("Precision: " + str(metrics.precision_score(y_test, y_predict_test)))

        class_names = ['ham', 'spam']
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay(NB_classifier, X_test, y_test,
                                          display_labels=class_names,
                                          cmap=plt.cm.Blues,
                                          normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
        plt.show()

        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]
        # predict probabilities
        lr_probs = NB_classifier.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Naive Bayes: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Naive Bayes')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()

        return

    def apply_svm(self, X, y):
        # DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # Training model
        # 'linear', 'poly', 'rbf'
        params = {'kernel': 'linear', 'C': 2, 'gamma': 1}
        svm_cv = svm.SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'], probability=True)
        svm_cv.fit(X_train, y_train)
        # Predicting the Test set results
        y_predict_test = svm_cv.predict(X_test)
        cm = confusion_matrix(y_test, y_predict_test)
        # sns.heatmap(cm, annot=True)
        # Evaluating Model
        print(classification_report(y_test, y_predict_test))
        print("test set")

        print("\nAccuracy Score: " + str(metrics.accuracy_score(y_test, y_predict_test)))
        print("F1 Score: " + str(metrics.f1_score(y_test, y_predict_test)))
        print("Recall: " + str(metrics.recall_score(y_test, y_predict_test)))
        print("Precision: " + str(metrics.precision_score(y_test, y_predict_test)))

        class_names = ['ham', 'spam']
        titles_options = [("Confusion matrix, without normalization", None),
                          ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay(svm_cv, X_test, y_test,
                                          display_labels=class_names,
                                          cmap=plt.cm.Blues,
                                          normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
        plt.show()

        # generate a no skill prediction (majority class)
        ns_probs = [0 for _ in range(len(y_test))]
        # predict probabilities
        lr_probs = svm_cv.predict_proba(X_test)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(y_test, ns_probs)
        lr_auc = roc_auc_score(y_test, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('SVM: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='SVM')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot
        pyplot.show()
        return


data_obj = data_read_write(r"C:\Users\Hassan\Desktop\ML I\NLP\Spam_Detections\emails.csv")
data_frame = data_obj.read_csv_file(r"C:\Users\Hassan\Desktop\ML I\NLP\Spam_Detections\processed.csv")
data_frame.head()
data_frame.tail()
data_frame.describe()
data_frame.info()
data_frame.head()
# Visualize dataset
# Let's see which message is the most popular ham/spam message
data_frame.groupby('spam').describe()

# Let's get the length of the messages
data_frame['length'] = data_frame['text'].apply(len)
data_frame['length'].max()

data_frame['length'].plot(bins=100, kind='hist')
# Length of characters for ham emails is more as compared to spam emails
sns.set(rc={'figure.figsize': (11.7, 8.27)})
ham_messages_length = data_frame[data_frame['spam'] == 0]
spam_messages_length = data_frame[data_frame['spam'] == 1]

ham_messages_length['length'].plot(bins=100, kind='hist', label='Ham')
spam_messages_length['length'].plot(bins=100, kind='hist', label='Spam')
# sns.distplot(ham_messages_length['length'], bins=10, norm_hist = True, label = 'Ham')
# sns.distplot(spam_messages_length['length'], bins=10, norm_hist = True, label = 'Spam')
plt.title('Distribution of Length of Email Text')
plt.xlabel('Length of Email Text')
plt.legend()

# ax = sns.distplot(ham_words_length, norm_hist = True, bins = 30, label = 'Ham')
# ax = sns.distplot(spam_words_length, norm_hist = True, bins = 30, label = 'Spam')

# plt.legend()
# plt.title('Distribution of Number of Words')
# plt.xlabel('Number of Words')
# plt.show()
