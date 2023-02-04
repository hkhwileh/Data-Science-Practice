from __future__ import absolute_import, division , print_function, unicode_literals

import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
from tensorflow import  keras as ks
from tensorflow.estimator import LinearRegressor
from sklearn import  datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print(tf.__version__)

boston_load = datasets.load_boston()
feature_columns = boston_load.feature_names
target_column = boston_load.target
boston_data = pd.DataFrame(boston_load.data,columns=feature_columns)

boston_data['MEDV'] = target_column.astype(np.float32)
print(boston_data.head())
sb.pairplot(boston_data, diag_kind="kde")
print(sb.pairplot(boston_data, diag_kind="kde"))


correlation_data = boston_data.corr()
correlation_data.style.background_gradient(cmap='coolwarm', axis=None)
plt.show()

stats = boston_data.describe()
boston_stats = stats.transpose()
print(boston_stats)

X_data = boston_data[[i for i in boston_data.columns if i not in ['MEDV']]]
Y_data = boston_data[['MEDV']]

training_feature , test_feature , training_labels, test_labels = train_test_split(X_data,Y_data,test_size=0.2)
print("No. of rows in Training Features:", training_feature.shape[0])
print("No. of rows in the test features:", test_feature.shape[0])

print("No. of columns in Training Feature:", training_feature.shape[1])
print("No. of columns in test features :", test_feature.shape[1])

print("No. of rows in training labels", training_labels.shape[0])
print("No. of rows in test labels", test_labels.shape[0])

print("No. of columns in training label", training_labels.shape[1])
print("No. of columns in test label", test_labels.shape[1])

def norm(x):
    stats = x.describe()
    stats =stats.transpose()
    return (x - stats['mean'])/ stats['std']

normed_train_features = norm(training_feature)
normed_test_features = norm(test_feature)

def feed_input(features_dataframe, target_dataframe, num_of_epochs=10,shuffle=True,batch_size=32):
    def input_feed_function():
        dataset = tf.data.Dataset.from_tensor_slices((dict(features_dataframe),target_dataframe))
        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size).repeat(num_of_epochs)
        return dataset
    return input_feed_function

train_feed_input = feed_input(normed_train_features,training_labels)
training_feed_input_testing = feed_input(normed_train_features,training_labels,num_of_epochs=1,shuffle=False)

test_feed_input = feed_input(normed_test_features, test_labels, num_of_epochs=1,shuffle=False)

feature_columns_numaric = [tf.feature_column.numeric_column(m) for m in training_feature.columns]

linear_model = LinearRegressor(feature_columns=feature_columns_numaric,optimizer='RMSProp')
linear_model.train(train_feed_input)

train_predictions = linear_model.predict(training_feed_input_testing)
test_predictions = linear_model.predict(test_feed_input)
train_predictions_series = pd.Series([p['predictions'][0] for p in train_predictions])
test_predictions_series = pd.Series([p['predictions'][0] for p in test_predictions])


train_predictions_df = pd.DataFrame(train_predictions_series, columns=['predictions'])
test_predictions_df = pd.DataFrame(test_predictions_series,columns=['predictions'])
training_labels.reset_index(drop=True,inplace=True)

train_predictions_df.reset_index(drop=True, inplace=True)
test_labels.reset_index(drop=True, inplace=True)
test_predictions_df.reset_index(drop=True, inplace=True)
train_labels_with_predictions_df = pd.concat([training_labels, train_predictions_df], axis=1)
test_labels_with_predictions_df = pd.concat([test_labels,test_predictions_df], axis=1)

def calculate_errors_and_r2(y_true, y_pred):
    mean_squared_err = (mean_squared_error(y_true, y_pred))
    root_mean_squared_err = np.sqrt(mean_squared_err)
    r2 = round(r2_score(y_true, y_pred)*100,0)
    return mean_squared_err, root_mean_squared_err, r2

train_mean_squared_error, train_root_mean_squared_error,train_r2_score_percentage = calculate_errors_and_r2(training_labels, train_predictions_series)

test_mean_squared_error, test_root_mean_squared_error,test_r2_score_percentage = calculate_errors_and_r2(test_labels, test_predictions_series)

print('Training Data Mean Squared Error = ', train_mean_squared_error)

print('Training Data Root Mean Squared Error = ', train_root_mean_squared_error)
print('Training Data R2 = ', train_r2_score_percentage)

print('Test Data Mean Squared Error = ', test_mean_squared_error)


print('Test Data Root Mean Squared Error = ', test_root_mean_squared_error)

print('Test Data R2 = ', test_r2_score_percentage)
