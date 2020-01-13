import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv("train.csv")
print(df_train)

df_test = pd.read_csv("test.csv")
print(df_test)

id_person = df_train.iloc[: 0].values
X = df_train.iloc[:, 1:-1].values
y = df_train.iloc[:, -1].values

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
Accuracy=[]

for train_indices, test_indices in k_fold.split(X):
    X_train, X_test, y_train, y_test = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    Accuracy.append(accuracy_score(y_test, y_test_pred))


print("Average 10-fold cross-validation accuracy=", np.mean(np.array(Accuracy)))

from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_n = scalerX.transform(X_train)

np.set_printoptions(suppress=True)
print("Original", X_train[0,:])
print("\n Transformed", X_train_n[0, :])
print("\n What StandardScaler really does:", ((X_train-np.mean(X_train, axis=0))/np.std(X_train, axis=0))[0,:] )


k_fold = KFold(n_splits=10,  shuffle=True, random_state=42)
Accuracy=[]
for train_indices, test_indices in k_fold.split(X):
    #print('Train: %s | test: %s' % (train_indices, test_indices))
    X_train, X_test, y_train, y_test =X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    #(A) data pre-processing 
    scalerX = StandardScaler()
    scalerX.fit(X_train)
    X_train_n = scalerX.transform(X_train)
    X_test_n  = scalerX.transform(X_test )
    
    #(B)  ML algorithm fitting
    clf = LogisticRegression(random_state=0, solver='lbfgs', C=1)
    clf.fit(X_train_n, y_train)    
    # (C) Prediction
    y_test_pred = clf.predict(X_test_n)    
    # (D) Score 
    Accuracy.append(accuracy_score(y_test,y_test_pred))
    
print("Average 10-fold cross-validation accuracy=",np.mean(np.array(Accuracy)))





