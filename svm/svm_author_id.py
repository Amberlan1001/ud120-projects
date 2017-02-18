#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

clf = SVC(kernel='rbf',C=10000.)
t0 = time()
clf.fit(features_train, labels_train)
print 'training time:',round(time() - t0, 3),'s'

# print clf.score(features_test, labels_test)
predictions = clf.predict(features_test)
# print predictions[10]
# print predictions[26]
# print predictions[50]

count = 0
for result in predictions:
    if result == 1:
        count = count + 1
print count

#########################################################

'''
training time: 207.594 s
0.984072810011

training time: 0.124 s
0.884527872582

training time: 0.139 s
0.616040955631

training time: 0.127 s
0.892491467577

training time: 137.599 s
877

'''
