import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv("email_clean.csv")
print("number of rows ", dataset.shape[0])
print("number of rows ", dataset.shape[1])

# splitting input and output
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=7)

clf = MultinomialNB()
clf.fit(x_train, y_train)

prediction_train = clf.predict(x_train)
prediction_test  = clf.predict(x_test)

print("Accuracy: " + str(accuracy_score(y_train, prediction_train)))
print()


print("Accuracy: "+ str(accuracy_score(y_test, prediction_test)))
print()

conf_mat = confusion_matrix(y_test, prediction_test)
print("Confusion Matrix")
print(conf_mat)

np.set_printoptions(suppress=True)
proba = clf.predict_proba(x_test)


def new_prodict(proba, threshold):
    classes = ["ham", "spam"]
    output = []
    for i in range(proba.shape[0]):
        indmax = np.argmax(proba[i, :])
        if proba[i, indmax]>threshold:
            output.append(classes[indmax])
        else:
            output.append("none")
    return output

ypred_new = new_prodict(proba, 0.8)
ind       = np.where(np.array(ypred_new)!= "none")
print(accuracy_score(np.array(ypred_new)[ind], y_test[ind]))













