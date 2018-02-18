import pandas as pd
df = pd.read_csv("file:///Users/bhavanarama/Desktop/train.csv");
X = pd.DataFrame()
X = df
X.dropna(axis=0)
y = X['LEAVE']
X = X.drop(['LEAVE'], axis=1)
X['COLLEGE'] = pd.get_dummies(X.COLLEGE)['one']
df1 = pd.read_csv("file:///Users/bhavanarama/Desktop/test.csv");
testset = pd.DataFrame()
testset = df1
testset['COLLEGE'] = pd.get_dummies(testset.COLLEGE)['one']
dummies = pd.get_dummies(X)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dummies = scaler.fit_transform(dummies)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty ='l1', C=1)
model.fit(dummies,y)



/// testing accuracy
import pandas as pd
df = pd.read_csv("file:///Users/bhavanarama/Desktop/train.csv");
X = pd.DataFrame()
X = df
X.dropna(axis=0)
y = X['LEAVE']
X = X.drop(['LEAVE'], axis=1)
X['COLLEGE'] = pd.get_dummies(X.COLLEGE)['one']
df1 = pd.read_csv("file:///Users/bhavanarama/Desktop/test.csv");
testset = pd.DataFrame()
testset = df1
testset['COLLEGE'] = pd.get_dummies(testset.COLLEGE)['one']
dummies = pd.get_dummies(X)
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dummies = scaler.fit_transform(dummies)
X_train, X_test, y_train, y_test = train_test_split(dummies, y, test_size=0.2,random_state=2)
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty ='l1', C=4)
model.fit(X_train, y_train)
accuracy_score(y_test, model.predict(X_test))
