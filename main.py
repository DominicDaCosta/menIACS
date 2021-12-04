#Co-Authors: Yesenia Andrade (BSCS 2024), Ali Sareini (BSCS 2024), Dominic DaCosta (BSCS 2024)

import pandas as pd
vaData = pd.read_csv("/content/sample_data/Virginia Data.csv")
vaData.head(20)

vaData.describe()
# vaData["2021 Pop."]
X, y = vaData["Name"], vaData["Growth"]

#Splitting Data into Training and Testing sets
from sklearn.model_selection import train_test_split
# Split the data into train set (80%) and validation set (20%)

train_set, validation_set = train_test_split(vaData, test_size=0.2, random_state=42)

X_train = train_set["Name"]
y_train = train_set["Growth"]

X_valid = validation_set["Name"]
y_valid = validation_set["Growth"]

#Extracting
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
# Use Fit_transform function to compute the parameters and then do the transformation
X_train = cv.fit_transform(X_train)
# Note that we use transform here because we want to use the same parameters learned from training data
X_valid = cv.transform(X_valid)
#GDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)

#Sequential Modeling w/ Atrificial Neural Networks
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
indices_one = X_train == 'Stagnant'
indices_two = X_train == 'Growing'
X_train[indices_one] = -1
X_train[indices_two] = 1
X_train

import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=vaData["Population Percent Change"]

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

#Final Production
X, y = vaData["Growth"], vaData["Population Percent Change"]
X.shape
y.shape

from sklearn.model_selection import train_test_split
# Split the data into train set (80%) and validation set (20%)
train_set, validation_set = train_test_split(vaData, test_size=0.2, random_state=42)

X_train = train_set["Population Percent Change"]
y_train = train_set["Growth"]

X_valid = validation_set["Population Percent Change"]
y_valid = validation_set["Growth"]



#GDClassifier

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
X_train = X_train.reshape(1,-1)
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train)
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
