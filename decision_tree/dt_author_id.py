#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
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


print len(features_test[0])
print features_test.shape

#########################################################
## your code goes here ###
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=40)
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
pred = clf.predict(features_test)
t2 = time()
print "Time taken to train", t1-t0
print "Time taken to predict", t2-t1
print "Time taken to overall", t2-t0

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print acc
########################################################


