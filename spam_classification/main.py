import re
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix

#import nltk
#nltk.download('stopwords')


df = pd.read_csv('spam.csv', encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis='columns')
print(df)

stemmer = SnowballStemmer('english')
stop = set(stopwords.words('english'))

df['v2'] = [re.sub('[^a-zA-Z]', ' ', sms) for sms in df['v2']]
word_list = [sms.split() for sms in df['v2']]
def normalize(words):
    current_words = list()
    for word in words:
        if word.lower() not in stop:
            updated_word = stemmer.stem(word)
            current_words.append(updated_word.lower())
    return current_words

word_list = [normalize(word) for word in word_list]
df['v2'] = [" ".join(word) for word in word_list]

print(df)
x_train, x_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=7)

cv = CountVectorizer()

x_train_df = cv.fit_transform(x_train)
print("number of emails=", x_train_df.shape[0])
print("number of  words=", x_train_df.shape[0])
x_test_df = cv.transform(x_test)

print(x_train_df)

row_index = 0
print(x_train_df[row_index, :].todense().shape)
print("this is the non-sparse matrix=", x_train_df[row_index, :].todense())
ind = np.where(x_train_df[row_index, :].todense()[0, :]>0)[1]
print()

print(x_train.values[row_index])
print()

print(cv.inverse_transform(x_train_df[row_index, :].todense()))
print()

print(ind)
print()

print(x_train_df[row_index, ind].todense())



#step 4
clf = MultinomialNB()
clf.fit(x_train_df, y_train)
prediction_train = clf.predict(x_train_df)
prediction_test  = clf.predict(x_test_df)

#step 5
print("Accuracy:" + str(accuracy_score(y_train, prediction_train)))
print()


#scores
print("Accuracy:" + str(accuracy_score(y_test, prediction_test)))
print()

conf_mat = confusion_matrix(y_test, prediction_test)
print("Confusion Matrix")
print(conf_mat)


import scipy.sparse as sc

i=0
ind = sc.find(x_train_df[i, :]>0)[1]
print("Indexes of non-zeroes elements=", ind)
x_train_df[0, ind].todense()

ind = sc.find(x_test_df[i, :]>0)[1]
print("indexes of non-zeroes elements=", ind)
x_test_df[0, ind].todense()








