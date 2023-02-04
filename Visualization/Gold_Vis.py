import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

gold_data = pd.read_csv(r"C:\Users\Hassan\Desktop\ML I\dataset\gld_price_data.csv")

print(gold_data.head())
print(gold_data.info())
correlation = gold_data.corr()

plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap="Blues")
plt.show()

print(correlation['GLD'])
sns.displot(correlation['GLD'], color='green')
plt.show()

X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']
print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=2)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100)

# training the model
regressor.fit(X_train, Y_train)

# prediction on test data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# R squared error
from sklearn import metrics

error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error :", error_score)

acc = metrics.accuracy_score(Y_test, test_data_prediction)
print("accurcy  :", acc)
