# -*- coding: UTF-8 -*-
#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.metrics import accuracy_score
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

# naive bayes
if False:
    from sklearn.naive_bayes import GaussianNB
    clf_nb = GaussianNB()
    clf_nb.fit(features_train, labels_train)
    pred_nb = clf_nb.predict(features_test)
    print 'Accuracy naive bayes:',accuracy_score(labels_test, pred_nb)
    # 0.884

# svm
if False:
    from sklearn.svm import SVC
    clf_svm = SVC()
    clf_svm.fit(features_train, labels_train)
    pred_svm = clf_svm.predict(features_test)
    print 'Accuracy svm:',accuracy_score(labels_test, pred_svm)
    # 0.92

# decision tree
if False:
    from sklearn import tree
    clf_dt = tree.DecisionTreeClassifier()
    clf_dt.fit(features_train, labels_train)
    pred_dt = clf_dt.predict(features_test)
    print 'Accuracy decision tree',accuracy_score(labels_test, pred_dt)
    # 0.908

# k nearest neighbors
if False:
    from sklearn.neighbors import KNeighborsClassifier
    clf_knn = KNeighborsClassifier(n_neighbors=4)
    clf_knn.fit(features_train, labels_train)
    pred_knn = clf_knn.predict(features_test)
    print 'Accuracy kNN:', accuracy_score(labels_test, pred_knn)
    # 0.94

# random forest
if False:
    from sklearn.ensemble import RandomForestClassifier
    clf_rf = RandomForestClassifier(n_estimators=15, min_samples_split=6)
    clf_rf.fit(features_train, labels_train)
    clf_rf = clf_rf.predict(features_test)
    print 'Accuracy random forests:', accuracy_score(labels_test, clf_rf)
    # 0.928

# adaboost
if False:
    from sklearn import ensemble
    clf_adb = ensemble.AdaBoostClassifier(n_estimators=15)
    clf_adb.fit(features_train, labels_train)
    pred_adb = clf.predict(features_test)
    acc_adb = accuracy_score(labels_test, pred_adb)
    print 'Accuracy of AdaBoost: %f' % acc_adb
    # 0.928

# 实际选择的算法
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=4)

clf.fit(features_train, labels_train)
t0 = time()
pred = clf.predict(features_test)
print 'Accuracy:',accuracy_score(labels_test, pred)
print 'predict and count accuracy time:',round(time() - t0, 3),'s'

t1 = time()
print 'Accuracy:',clf.score(features_test, labels_test)
print 'count accuracy time:',round(time() - t1, 3),'s'

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass