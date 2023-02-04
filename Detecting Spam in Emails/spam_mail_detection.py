import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Data Collection and Pre-processing

raw_mail_data = pd.read_csv(r"C:\Users\Hassan\Desktop\ML I\NLP\Spam_Detections\dataset\mail_data.csv")
# print(raw_mail_data.head())

# replace null value with null string
raw_mail = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
# print(raw_mail.head())

# checking the number of columns and rows in data frame
# print(raw_mail.shape)

# Label spam mail as 0 : harm , 1:spam
raw_mail['Category'] = raw_mail['Category'].replace('ham', 0)
raw_mail['Category'] = raw_mail['Category'].replace('spam', 1)
# or you can use the following
# raw_mail.loc[raw_mail['Category'] == 'spam', 'Category',] = 1
# raw_mail.loc[raw_mail['Category'] == 'ham', 'Category',] = 0
print(raw_mail.head())

# separating the data as texts and label
X = raw_mail['Message']
y = raw_mail['Category']
# Splitting the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)

# transform the text data to feature vectors that you can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert y_train & y_test values as integers value
y_train = y_train.astype('int')
y_test = y_test.astype('int')
'''
print('----------------------X_train--------------------',X_train)
print('----------------------X_test---------------------',X_test)
print('----------------------y_train--------------------',y_train)
print('----------------------y_test--------------------',y_test)
print('----------------------X_test_features-----------',X_test_features)
'''
# Logistic Regression
model = LogisticRegression()
# Training Logistic Regression
model.fit(X_train_features, y_train)
# Evaluating the trained model

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

print('Accuracy on Training Data', accuracy_on_training_data)

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

print('Accuracy on Training Data', accuracy_on_test_data)

# Building a predective mail 
input_mail = input("Please Enter A Mail text to check if it's spam or harm : ")

input_mail_features = feature_extraction.transform([str(input_mail)])

out = model.predict(input_mail_features)
print(out)
if out[0] == 0:
    print("Ham Mail")
else:
    print("Spam Mail")
