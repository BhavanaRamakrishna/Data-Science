import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import csv
import numpy

df = pd.read_csv("train.csv");


X = pd.DataFrame()
X = df

%matplotlib inline
import matplotlib.pyplot as plt
num_bins = 10
X.hist(bins=num_bins, figsize=(20,15))
plt.savefig("hr_histogram_plots")
plt.show()

cat_vars=["COLLEGE","REPORTED_SATISFACTION","REPORTED_USAGE_LEVEL","CONSIDERING_CHANGE_OF_PLAN"]
for var in cat_vars:
	cat_list='var'+'_'+var
	cat_list = pd.get_dummies(X[var], prefix=var)
	X1=X.join(cat_list)
	X=X1

X.drop(X.columns[[0,8,9,10]], axis=1, inplace=True)
X_cols=X.columns.values.tolist()
y=X['LEAVE']
X=X.drop('LEAVE', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

logreg = DecisionTreeClassifier(min_samples_split=18, min_samples_leaf=8, criterion='entropy')
logreg.fit(X_train, y_train)
print accuracy_score(y_test, logreg.predict(X_test))
