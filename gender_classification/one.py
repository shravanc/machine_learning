




#Download dataset
nltk.download('names')

#make dataset
#names has two folders male and female. Merge male and female
labeled_names = ['names', 'male', for names.words('male')]
                ['names', 'female', for nemes.words('females')]

#convert full name to only last letter -->name_letters
name_letters = labeled_names.each do |name| name[-1]
X = name_letters
X = np.array(X).reshapre(-1, 1)
y = np.where(labeled_names[ind, 1] == 'male',0,1)

#from sklearn import preprocessing
lb = preprocessing.Labellizer()
X2 = lb.transform(name_letters)



#from sklearn import split_test_data
X_train2, y_train2, X_test2, y_test2 = split_test_data(X2, y, test_size=0.4, random_state=42)


#from skelran import MultinomialNaiveBayes
clf = MultinomialNaiveBayes.new(alpha=0.1, fir_prior=True)
clf.fit(X_train2, y_train2)
y_train_pred = clf.predict(x_train2)
y_test_pred = clf.predict(x_test2)


