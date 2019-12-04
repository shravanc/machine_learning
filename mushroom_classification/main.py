import pandas as pd
import numpy as np

data = np.array([
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [0, 0, 2],
    [0, 1, 0],
    [0, 0, 2]])

df = pd.DataFrame(data, columns=['EdibleOrPoisonous', 'RedColor', 'CapSurface'])
print(df)

train_df = df.copy()
col = np.array(['RedColor', 'CapSurface'])
for f in range(1, df.shape[1]):
    for elem in df.iloc[:, f].unique():
        train_df[col[f-1]+'_'+str(elem) ] = (train_df.iloc[:, f]==elem)+0.0

train_df = train_df.drop(columns=col)
print(train_df)


X = train_df.iloc[:, 1:].values
y = train_df.iloc[:, 0].values

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0)
clf.fit(X, y)
clf.predict_proba(np.array( [[1,0,1,0,0]]  ))

clf = MultinomialNB(alpha=0)
clf.fit(X, y)
p = clf.predict_proba(np.array([[1,0,0,1,0]]))
print(p)

from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB(alpha=1)
clf.fit(X, y)
p = clf.predict_proba(np.array( [[1,0,0,1,0]]  ) )
print(p)


#*********************REAL Example*****************
df = pd.read_csv('./mushrooms.csv')
from sklearn.utils import shuffle

df = shuffle(df, random_state=42)
print(df)


train_df = df[:7000]
test_df  = df[7000:]

print(train_df)


# from this we can derive the accuracy of the majority class classifier
print(train_df['class'].value_counts(normalize=1))

target = []
for i in range(len(train_df['class'].values)):
    if train_df['class'].values[i]=='e':
        target.append(0)
    if train_df['class'].values[i]=='p':
        target.append(1)
    if train_df['class'].values[i]=='u':
        target.append(2)

target = np.array(target)
print(target)

del train_df['class']

#we transform inputs for multinomialNB
cols = list(train_df)
print("cols--->", cols)
for f in cols:
    for elem in df[f].unique():
        train_df[f+'_'+str(elem)] = (train_df[f]==elem)

#we delete old columns
print("Before deleting---->", train_df)
for f in cols:
    del train_df[f]
print("after deleting----->", train_df)
train_df.head()
train_df=train_df+0.0
print("after adding 0.0.----->", train_df)


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
train_x = train_df.values
clf.fit(train_x, target)

from sklearn.metrics import accuracy_score
y_pred = clf.predict(train_x)
a = accuracy_score(y_pred, target)
print(a)


test_y = test_df['class']
del test_df['class']
for f in cols:
    for elem in df[f].unique():
        test_df[f+'_'+str(elem)] = (test_df[f]==elem)

for f in cols:
    del test_df[f]

print(test_df)

test_x = test_df.values

test_y1=[]
for i in range(len(test_y)):
    if test_y.values[i] == 'e':
        test_y1.append(0)
    if test_y.values[i] == 'p':
        test_y1.append(1)
    if test_y.values[i] == 'u':
        test_y1.append(2)

test_y1 = np.array(test_y1)

y_pred = clf.predict(test_x)
accuracy_score(y_pred, test_y1)

from sklearn.metrics import confusion_matrix
CC = confusion_matrix(test_y1, y_pred)
print(CC)






