from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

"""
Andrew McCann
CS445 Machine Learning
Homework #3
Melanie Mitchell

THINGS TODO:
General:
    DONE: Split data to test/training (equal +/- examples)
    DONE: Find out what format SVC requires
        DONE: Format data for SVC with python

Maybe:
    Implement methods of isolating data transformation

Experiment #1:
    Linear Kernel
    10-fold cross validation
        Used to test C values from 0 to 1 with .1 increment.
        Get largest C value from comparing accuracy (Argmax + array indeces should make trivial)
    Train a brand new SVM using this c param.
    Test on SVM model.
    Create ROC curve with 200 evenly spaced threshholds (wtf?)

Experiment #2
    Use SVM Model from #1
    Obtain weight vector w.
    From m=2 to 57
    Select features in varying number with highest weight
        Train a linear SVM on all training data using only m features and C from ex1
        Test this SVM on test data to pull accuracy

    Plot graph of accuracy vs. m

Experiment #3
    Same as two, but select the features at random, not the highest.

"""



def experiment_one(folded_array):
    """
    Experiment #1:
    Linear Kernel
    10-fold cross validation
        Used to test C values from 0 to 1 with .1 increment.
        Get largest C value from comparing accuracy (Argmax + array indeces should make trivial)
    Train a brand new SVM using this c param.
    Test on SVM model.
    Create ROC curve with 200 evenly spaced threshholds (wtf?)
    :param folded_array:
    :return:
    """
    accuracy = []
    c_vals = np.arange(0, 1.1, 0.1)
    print(c_vals)
    for i in range(len(folded_array)):
        np.random.shuffle(folded_array[i])
        for j in range(len(c_vals)):
            print(c_vals[j])
            svm = SVC(C=j, kernel='linear')
            #svm.fit(folded_array[i][:,:-1], folded_array[i][:, -1])
            print(folded_array[i][:,:-1])
            print(folded_array[i][:, -1])


def cross_validate(training_set, k):
    size = len(training_set)//k
    folds = []

    for i in range(k-1):
        folds.append(training_set[(i*size):((i+1)* size)])
    folds.append(training_set[((k-1)*size)])

    return folds


def LoadSpamData(filename="spambase.data"):
    """
    Each line in the datafile is a csv with features values, followed by a single label (0 or 1), per sample; one sample per line
    """
    raw_data = np.loadtxt(filename, delimiter=',')

    # Split raw data into examples.
    negatives = raw_data[raw_data[:, -1]== 0]
    positives = raw_data[raw_data[:, -1]== 1]

    # Grab lowest count
    lowest_value = min(len(negatives), len(positives))

    # Resave the values with the extra cut off.
    negatives = negatives[:lowest_value]
    positives = positives[:lowest_value]

    # To preserve the ratio of + and - we split prior to shuffling
    mid = lowest_value//2
    negatives1 = negatives[:mid]
    negatives2 = negatives[mid:]
    positives1 = positives[:mid]
    positives2 = positives[mid:]

    # Reform into training/test
    training_data = np.vstack((negatives1, positives1))
    test_data = np.vstack((negatives2, positives2))

    # Pulled from HW2 preprocessing
    # Grab values to normalize data
    scalar = preprocessing.StandardScaler().fit(training_data[:, :-1])

    # We do not want to scale the labels at the end, so skip that
    training_data[:, :-1] = scalar.transform(training_data[:, :-1])
    # Scale test data using training data values.
    test_data[:, :-1] = scalar.transform(test_data[:, :-1])


    return training_data, test_data


def main():
    training_set, test_set = LoadSpamData()
    folded = cross_validate(training_set, 10)
    experiment_one(folded)

    # features, labels = BalanceDataset(features, labels)
    # features, labels = ConvertDataToArrays(features, labels)
    # features = NormalizeFeatures(features)
    # indices = [0,1,2]
    # features = FeatureSubset(features, indices)
    # test_svm(features, labels)



if __name__ == "__main__":
    main()
