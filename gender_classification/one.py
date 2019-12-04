




#Download dataset
nltk.download('names')

#make dataset
#names has two folders male and female. Merge male and female
from nltk.corpus import names
labeled_names = ( [ ('names', 'male') for name in names.words('male.txt')] +\
                [('names', 'female') for name in names.words('female.txt')] )
random.shuffle(labeled_names)
labeled_names = np.array(labeled_names)

#convert full name to only last letter -->name_letters
name_letters = [gneder_features(name) for name in labeled_names[:, 0]]
name_letters = np.array(name_letters)
ind = np.where((name_letters!=' ')==True)[0]
name_letters = name_letters[ind]

X = name_letters

X = np.array(X).reshapre(-1, 1)
y = np.where(labeled_names[ind, 1] == 'male',0,1)

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(X)
X2 = lb.transform(X)



from sklearn.preprocessing import train_test_split
X_train2, y_train2, X_test2, y_test2 = train_test_split(X2, y, test_size=0.4, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

clf = MultinomialNB.new(alpha=0.1, fit_prior=True)
clf.fit(X_train2, y_train2)
y_train_pred  = clf.predict(x_train2)
y_test_pred   = clf.predict(x_test2)



