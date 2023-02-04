import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

raw_mail_data = pd.read_csv(r"C:\Users\Hassan\Desktop\ML I\NLP\Spam_Detections\dataset\mail_data.csv")

raw_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

raw_data['Category'] = raw_data['Category'].replace('ham', 0)
raw_data['Category'] = raw_data['Category'].replace('spam', 1)
print("-----------------correlation------------------------", raw_data.corr())

X = raw_data.Message
y = raw_data.Category
print('----------------------------X----------------------', X)
print('-----------------------------y---------------------', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# y_train = y_train.astype('int')
# y_test = y_test.astype('int')

model = LogisticRegression()
model.fit(X_train_features, y_train)

prediction_data = model.predict(X_train_features)

accuracy_data = accuracy_score(y_train, prediction_data)

print('Accuracy on Training Data', accuracy_data)
