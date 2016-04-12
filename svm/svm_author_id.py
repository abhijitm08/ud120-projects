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

def accuracy(pred, labels_test):
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)
    return acc

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#Training on only 1% of training sample as traning with svm takes ages (speed-accuracy tradeoff)
#eg. fraud detection, voice recogniser require speed compromise accuracy hence train quickly of less data
#########################################################
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
#########################################################

#########################################################
### your code goes here ###
from sklearn.svm import SVC

#For getting optimised value of C which is 10000.
#########################################################
#for Cs in [10.0, 100., 1000., 10000.]:
#    #clf = SVC(kernel="linear")
#    clf = SVC(C=Cs, kernel="rbf")
#    t0 = time()
#    clf.fit(features_train, labels_train)
#    t1 = time()
#    print "training time with C=", Cs, round(t1-t0, 3), "s"
#    pred = clf.predict(features_test)
#    t2 = time()
#    print "testing time with C=", Cs, round(t2-t1, 3), "s"
#    print "accuracy with C=", Cs, accuracy(pred, labels_test)
#########################################################

#Use full training set and optimised c parameter to get high accuracy even if time taken is large
#########################################################
clf = SVC(C=10000., kernel="rbf")
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
print "training time with C=", 10000., round(t1-t0, 3), "s"
pred = clf.predict(features_test)
t2 = time()
print "testing time with C=", 10000., round(t2-t1, 3), "s"
print "accuracy with C=", 10000., accuracy(pred, labels_test)
print "pred[10th]", pred[10]
print "pred[26th]", pred[26]
print "pred[50th]", pred[50]

print "Elements in test set", len(features_test)
chris = [name for name in pred if name==1]
sara = [name for name in pred if name==0]
true_chris = [name for name in labels_test if name==1]
true_sara = [name for name in labels_test if name==0]
print "True Chris are:", len(true_chris)
print "Pred Chris are:", len(chris)
print "True Sara are:", len(true_sara)
print "Pred Sara are:", len(sara)
#########################################################

