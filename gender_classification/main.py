from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import random
from collections import Counter
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier, classify
import pandas as pd

# get last letter
def gender_features(word):
    return word[-1].lower()

print(gender_features('Shravan'))

#downlaod names
nltk.download('names')

#makes a datasets
from nltk.corpus import names
labeled_names = ( [(name, 'male') for name in names.words('male.txt')] +\
                  [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)
labeled_names=np.array(labeled_names)
print(labeled_names[0:10, :])

name_letters = [gender_features(name) for name in labeled_names[:, 0]]

name_letters = np.array(name_letters)
ind = np.where((name_letters!=' ')==True)[0]
name_letters = name_letters[ind]

X = name_letters

X = np.array(X).reshape(-1, 1)
y = np.where(labeled_names[ind, 1] == 'male',0,1)

print("X-->", X)


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(X)
X2 = lb.transform(X)
print("X2-->", X2)

#splitting the dataset in training and testing
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.4, random_state=42)
print(X_train2.shape)
print(X_test2.shape)

# also splitting the original dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# general ML Recipe algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

clf = MultinomialNB(alpha=0.1, fit_prior=True)
clf.fit(X_train2, y_train2)
y_train_pred = clf.predict(X_train2)
y_test_pred  = clf.predict(X_test2)


np.set_printoptions(suppress=True)
print(clf.predict_proba(X_test2))
print("X_test-->", X_test)


p = clf.predict_proba(lb.transform(np.array( [[gender_features("Anna")]] )  ))
print("P-->", p)

print("accuracy training set ", accuracy_score(y_train_pred, y_train2))
print("accuracy test set     ", accuracy_score(y_test_pred, y_test2))

CC = confusion_matrix(y_test2, y_test_pred)
print(CC)


